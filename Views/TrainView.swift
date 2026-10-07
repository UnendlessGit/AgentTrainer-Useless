import SwiftUI
import Charts

struct TrainView: View {
    var session: AppSession
    private var trainer: TrainingCoordinator { session.trainer }
    private var store: WorkspaceStore { session.store }
    private var model: AIModel? { store.models.first(where: { $0.id == trainer.selectedModelID }) }

    var body: some View {
        @Bindable var trainer = trainer
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(title: "Train", subtitle: "Turn demonstrations into understanding, on your Mac.") {
                    StatusPill(title: trainer.progress.modelID == trainer.selectedModelID ? trainer.progress.phase.rawValue : "Ready",
                               color: trainer.isBusy ? .blue : .secondary)
                }
                Surface(title: "Training model", symbol: "cpu") {
                    Picker("Model", selection: $trainer.selectedModelID) {
                        Text("Select a model").tag(nil as UUID?)
                        ForEach(store.models) { Text($0.name).tag(Optional($0.id)) }
                    }.disabled(trainer.isBusy)
                    if let model { Text(model.compatibility).font(.caption).foregroundStyle(.secondary) }
                }
                HStack(alignment: .top, spacing: 20) {
                    stageCard(.pretraining, title: "1. Pre-train", symbol: "sparkles", description: "Learn how visual environments change in response to actions.")
                    stageCard(.imitation, title: "2. Train", symbol: "waveform.path", description: "Learn demonstrated actions, pointer locations and timing.")
                }
                if trainer.isBusy || trainer.progress.phase != .idle { progressPanel }
                Surface(title: "Training settings", symbol: "slider.horizontal.3") {
                    HStack(alignment: .top, spacing: 30) {
                        VStack(alignment: .leading, spacing: 14) {
                            Stepper("Epochs: \(trainer.settings.epochs)", value: $trainer.settings.epochs, in: 1...1000)
                            Picker("Batch size", selection: $trainer.settings.batchSize) { ForEach([1, 2, 4, 8], id: \.self) { Text("\($0) sequences").tag($0) } }
                            TextField("Learning rate", value: $trainer.settings.learningRate, format: .number.precision(.fractionLength(1...6)))
                                .accessibilityLabel("Learning rate").accessibilityIdentifier("training.learningRate")
                            TextField("Weight decay", value: $trainer.settings.weightDecay, format: .number.precision(.fractionLength(0...4)))
                                .accessibilityLabel("Weight decay").accessibilityIdentifier("training.weightDecay")
                        }
                        VStack(alignment: .leading, spacing: 14) {
                            TextField("Gradient clip", value: $trainer.settings.gradientClip, format: .number)
                                .accessibilityLabel("Gradient clip").accessibilityIdentifier("training.gradientClip")
                            Stepper("Checkpoint every \(trainer.settings.checkpointInterval) steps", value: $trainer.settings.checkpointInterval, in: 1...1000)
                            Picker("Held-out recordings", selection: $trainer.settings.validationFraction) {
                                Text("None · training loss only").tag(0.0); Text("20%").tag(0.2); Text("30%").tag(0.3)
                            }
                            Stepper("Run budget: \(trainer.settings.maximumRunMinutes) min", value: $trainer.settings.maximumRunMinutes, in: 1...30)
                        }
                    }.disabled(trainer.isBusy)
                    Toggle("Balance input choices against waits", isOn: $trainer.settings.balancesInputChoices).disabled(trainer.isBusy)
                    Text("Keeps the recorded wait/input frequency while giving rarer input choices more training weight. Applies to imitation learning.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Validation uses separate recordings. With one recording, no validation score is reported. A run pauses at its time budget; Resume restores its data split, optimizer and memory.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }.padding(30)
        }.task { if trainer.selectedModelID == nil { trainer.selectedModelID = store.models.first?.id } }
        .onDisappear { trainer.saveSettings() }
    }

    private func stageCard(_ stage: TrainingStage, title: String, symbol: String, description: String) -> some View {
        Surface(title: title, symbol: symbol) {
            Text(description).foregroundStyle(.secondary).frame(minHeight: 38, alignment: .topLeading)
            let count = model.map { trainer.recordings(for: $0, stage: stage).count } ?? 0
            Metric(title: "ASSIGNED ELIGIBLE RECORDINGS", value: count.formatted())
            HStack {
                Button(stage == .pretraining ? "Pre-train" : "Train") {
                    if let model { trainer.start(model: model, stage: stage) }
                }.buttonStyle(.borderedProminent).disabled(model == nil || count == 0 || !store.activeOperations.isEmpty || store.migrating)
                Button("Resume") { if let model { trainer.resume(model: model, stage: stage) } }
                    .disabled(model == nil || !store.activeOperations.isEmpty || store.migrating || (stage == .pretraining ? model?.pretrainedCheckpoint : model?.trainedCheckpoint) == nil)
            }
            Text(stage == .imitation && model?.pretrainingCompatible == true ? "New training starts from your pre-trained checkpoint." : "Starts from new weights; previous checkpoints are preserved.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var progressPanel: some View {
        let p = trainer.progress
        return Surface(title: p.stage == .pretraining ? "Pre-training activity" : "Training activity", symbol: "chart.xyaxis.line") {
            if let request = trainer.lastRequest {
                Text("Results for \(request.model.name)").font(.headline)
                if request.model.id != trainer.selectedModelID {
                    Text("These results belong to the previous run. Start training to see results for the selected model.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Text(p.message).foregroundStyle(p.phase == .failed ? Color.red : .secondary)
                Spacer()
                if trainer.isBusy {
                    Button("Pause", action: trainer.pause)
                    Button("Cancel", role: .destructive, action: trainer.cancel)
                }
            }
            if p.stepsPerEpoch > 0 {
                ProgressView(value: min(1, (Double(p.epoch) + Double(p.cursor) / Double(p.stepsPerEpoch)) / Double(max(1, p.epochs))))
            }
            HStack(spacing: 25) {
                Metric(title: "STEP", value: p.step.formatted())
                Metric(title: "EPOCH", value: "\(min(p.epoch + 1, p.epochs))/\(p.epochs)")
                Metric(title: "TRAINING LOSS", value: number(p.loss))
                Metric(title: "VALIDATION LOSS", value: number(p.validationLoss))
                Metric(title: "GRADIENT NORM", value: number(p.gradientNorm))
            }
            if !p.history.isEmpty {
                Chart(p.history) { point in
                    LineMark(x: .value("Step", point.step), y: .value("Loss", point.training)).foregroundStyle(by: .value("Series", "Training"))
                    if let validation = point.validation {
                        PointMark(x: .value("Step", point.step), y: .value("Loss", validation)).foregroundStyle(by: .value("Series", "Validation"))
                    }
                }.chartForegroundStyleScale(["Training": Color.blue, "Validation": Color.orange]).frame(height: 160)
            }
            if let evaluation = p.actionEvaluation {
                HStack(spacing: 25) {
                    Metric(title: "HELD-OUT ACTION ACCURACY", value: percentage(evaluation.accuracy))
                    Metric(title: "NON-WAIT ACCURACY", value: percentage(evaluation.nonWaitAccuracy))
                    Metric(title: "NON-WAIT PRECISION", value: percentage(evaluation.nonWaitPrecision))
                }
                Text("\(evaluation.total.formatted()) held-out decisions, including \(evaluation.nonWaitTotal.formatted()) non-wait actions. Evaluation uses recorded history; use Run to verify closed-loop behavior.")
                    .font(.caption).foregroundStyle(.secondary)
                if let rows = evaluation.actionBreakdown {
                    DisclosureGroup("Validation by action") {
                        Grid(alignment: .leading, horizontalSpacing: 30, verticalSpacing: 8) {
                            GridRow {
                                Text("Action"); Text("Correct / demonstrated"); Text("Predicted")
                            }.fontWeight(.medium)
                            ForEach(rows) { row in
                                GridRow {
                                    Text(row.token == 0 ? "Wait" : row.action.label)
                                    Text("\(row.correct) / \(row.targets)")
                                    Text(row.predictions.formatted())
                                }
                            }
                        }.font(.caption).padding(.top, 8).frame(maxWidth: .infinity, alignment: .leading)
                        Text("Key presses and releases are separate targets. Correct releases do not establish that the model chose the right key to press.")
                            .font(.caption).foregroundStyle(.secondary).padding(.top, 4).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            HStack(spacing: 25) {
                Metric(title: "STEPS / SECOND", value: String(format: "%.2f", p.stepsPerSecond))
                Metric(title: "MLX ACTIVE / CACHE", value: "\(bytes(p.activeMemory)) / \(bytes(p.cacheMemory))")
                Metric(title: "APP FOOTPRINT", value: p.physicalFootprint.map { bytes(Int($0)) } ?? "—")
                Metric(title: "CPU · 100% / CORE", value: String(format: "%.0f%%", p.cpuPercent))
            }
            Text("Learning rate \(String(format: "%.6f", p.learningRate)) · \(p.trainingExamples.formatted()) training decisions · \(p.validationRecordings) held-out recordings · \(p.excludedOutsideTarget) actions outside the capture target excluded.")
                .font(.caption).foregroundStyle(.secondary)
            if p.stage == .imitation && p.trainingExamples > 0 {
                Text("Training data: \(p.trainingNonWaitExamples.formatted()) input transitions and \((p.trainingExamples - p.trainingNonWaitExamples).formatted()) waits.")
                    .font(.caption).foregroundStyle(.secondary)
                if p.trainingNonWaitExamples == 0 {
                    Text("These recordings only teach waiting. Add action demonstrations to teach other behavior.").font(.caption).foregroundStyle(.orange)
                }
            }
            Text(p.checkpoint.map { "Latest checkpoint: \($0.uuidString.prefix(8)) · weights, optimizer and memory saved" } ?? "No checkpoint written yet.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Metal acceleration is active during tensor work. macOS does not expose a reliable per-app GPU utilization percentage here.")
                .font(.caption).foregroundStyle(.secondary)
            Text("App footprint includes unified-memory allocations. MLX active and cache memory are components of that total, not additional usage.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func number(_ value: Float?) -> String { value.map { String(format: "%.4f", $0) } ?? "—" }
    private func percentage(_ value: Double?) -> String { value.map { String(format: "%.1f%%", $0 * 100) } ?? "—" }
    private func bytes(_ value: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory) }
}

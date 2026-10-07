import Foundation
import MLX
import MLXNN
import MLXRandom
import Darwin

enum TrainingPhase: String, Sendable {
    case idle = "Ready", preparing = "Preparing data", training = "Training", validating = "Validating"
    case checkpointing = "Saving checkpoint", paused = "Paused", cancelled = "Cancelled", complete = "Complete", failed = "Failed"
    var isBusy: Bool { [.preparing, .training, .validating, .checkpointing].contains(self) }
}

struct TrainingProgress: Sendable {
    var phase: TrainingPhase = .idle
    var message = "Select a model and a training stage."
    var modelID: UUID?
    var stage: TrainingStage = .imitation
    var step = 0
    var epoch = 0
    var epochs = 0
    var cursor = 0
    var stepsPerEpoch = 0
    var trainingExamples = 0
    var validationRecordings = 0
    var excludedOutsideTarget = 0
    var loss: Float?
    var validationLoss: Float?
    var actionEvaluation: ActionEvaluation?
    var gradientNorm: Float?
    var learningRate: Float = 0
    var stepsPerSecond = 0.0
    var activeMemory = 0
    var cacheMemory = 0
    var residentMemory: UInt64 = 0
    var cpuPercent = 0.0
    var elapsedSeconds = 0.0
    var checkpoint: UUID?
    var history: [LossPoint] = []
}

struct LossPoint: Identifiable, Sendable {
    var step: Int
    var training: Float
    var validation: Float?
    var id: Int { step }
}

final class TrainingControl: @unchecked Sendable {
    enum Request { case run, pause, cancel }
    private let lock = NSLock()
    private var request: Request = .run
    func set(_ value: Request) { lock.withLock { request = value } }
    var current: Request { lock.withLock { request } }
}

struct TrainingRequest: Sendable {
    let model: AIModel
    let settings: TrainingSettings
    let stage: TrainingStage
    let items: [RecordingItem]
    let preferences: AppPreferences
    let resume: Bool
}

/// All MLX arrays and modules are confined to this one serial worker. Main-actor
/// UI receives value snapshots only; it never reads or mutates a live tensor.
enum TrainingWorker {
    static let queue = DispatchQueue(label: "com.agenttrainer.learning", qos: .userInitiated)

    static func run(_ request: TrainingRequest, control: TrainingControl,
                    publish: @escaping @Sendable (TrainingProgress) -> Void,
                    checkpointSaved: @escaping @Sendable (CheckpointManifest) -> Void) {
        let started = Date(), deadline = started.addingTimeInterval(Double(request.settings.maximumRunMinutes) * 60)
        var progress = TrainingProgress(phase: .preparing, message: "Checking the selected recordings…", modelID: request.model.id,
                                        stage: request.stage, epochs: request.settings.epochs, learningRate: request.settings.learningRate)
        publish(progress)
        let temporary = URL(fileURLWithPath: request.preferences.checkpointsPath).appendingPathComponent(".datasets").appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary); Memory.clearCache() }
        func checkPreparation() throws {
            if control.current != .run { throw CancellationError() }
            if Date() >= deadline { throw DataIntegrityError.invalidData("The run's time budget ended during data preparation. Use fewer recordings for this run.") }
        }
        do {
            try request.settings.validate()
            try request.model.configuration.validate()
            let estimate = request.model.configuration.estimatedWorkingSetBytes(batchSize: request.settings.batchSize)
            guard estimate < request.preferences.memoryLimitGB * 1_073_741_824 * 3 / 4 else {
                throw DataIntegrityError.invalidData("This architecture and batch are estimated to exceed the selected memory budget. Reduce batch size or sequence length, or increase the MLX memory limit in Settings.")
            }
            Memory.memoryLimit = request.preferences.memoryLimitGB * 1_073_741_824
            Memory.cacheLimit = request.preferences.cacheLimitGB * 1_073_741_824
            let dataset = try PreparedDataset.prepare(items: request.items, configuration: request.model.configuration,
                settings: request.settings, stage: request.stage, root: temporary, checkCancellation: checkPreparation) { message in
                    progress.message = message; publish(progress)
                }
            progress.trainingExamples = dataset.exampleCount
            progress.validationRecordings = dataset.validation.count
            progress.excludedOutsideTarget = dataset.excludedOutsideTarget
            MLXRandom.seed(request.settings.seed)
            let model = PolicyNetwork(configuration: request.model.configuration)
            let optimizer = ResumableAdamW(learningRate: request.settings.learningRate, weightDecay: request.settings.weightDecay)
            let checkpoints = CheckpointStore(root: URL(fileURLWithPath: request.preferences.checkpointsPath))
            var epoch = 0, cursor = 0, step = 0
            var hidden: [MLXArray] = [], lastLoss: Float = 0, validationLoss: Float?, bestLoss: Float?
            if request.resume {
                guard let (saved, directory) = try checkpoints.latest(modelID: request.model.id, stage: request.stage, configuration: request.model.configuration) else {
                    throw DataIntegrityError.invalidData("There is no resumable checkpoint for this model and stage.")
                }
                guard saved.datasetFingerprint == dataset.fingerprint, saved.settings == request.settings else {
                    throw DataIntegrityError.invalidData("The data or training settings changed since this checkpoint. Restore them to resume exactly, or start a new training run.")
                }
                try restoreWeights(model, from: directory)
                let state = try MLX.loadArrays(url: directory.appendingPathComponent("optimizer.safetensors"))
                try optimizer.restore(state, model: model)
                if model.configuration.memory == .recurrent && saved.sampleCursor > 0 {
                    hidden = try (0..<model.configuration.memoryDepth).map { index in
                        guard let value = state["carry.\(index)"] else { throw DataIntegrityError.invalidData("This checkpoint is missing recurrent state.") }
                        return value
                    }
                }
                epoch = saved.epoch; cursor = saved.sampleCursor; step = saved.step
                lastLoss = saved.trainingLoss; validationLoss = saved.validationLoss; bestLoss = saved.bestValidationLoss
                progress.actionEvaluation = saved.actionEvaluation
                progress.checkpoint = saved.id
                guard epoch < request.settings.epochs else { throw DataIntegrityError.invalidData("This run already completed all epochs. Start a new run to train again.") }
            } else if request.stage == .imitation && request.model.pretrainingCompatible {
                guard let identifier = request.model.pretrainedCheckpoint, let id = UUID(uuidString: identifier) else {
                    throw DataIntegrityError.invalidData("The assigned pre-training checkpoint identifier is invalid.")
                }
                let (saved, directory) = try checkpoints.load(modelID: request.model.id, checkpointID: id, configuration: request.model.configuration)
                guard saved.stage == .pretraining else { throw DataIntegrityError.invalidData("The assigned pre-training checkpoint has the wrong stage.") }
                try restoreWeights(model, from: directory)
                progress.message = "Continuing from pre-trained representations."
            }
            eval(model)
            model.train(true)
            let stage = request.stage
            let lossGradient = valueAndGrad(model: model) { model, arrays in PolicyLoss.values(model, arrays, stage: stage) }
            var schedule = SequenceSchedule(recordings: dataset.training, batchSize: request.settings.batchSize,
                sequenceLength: model.configuration.sequenceLength, seed: request.settings.seed &+ UInt64(epoch))
            guard (0...schedule.count).contains(cursor), epoch >= 0, optimizer.step == step else {
                throw DataIntegrityError.invalidData("The checkpoint cursor or optimizer step is inconsistent.")
            }
            if let plan = schedule.plan(at: cursor), !plan.resetsMemory, model.configuration.memory == .recurrent {
                guard hidden.count == model.configuration.memoryDepth,
                      hidden.allSatisfy({ $0.shape == [plan.recordings.count, model.configuration.memorySize] }) else {
                    throw DataIntegrityError.invalidData("The checkpoint's recurrent state has incompatible dimensions.")
                }
            }
            var sampleTime = Date(), sampleCPU = processCPUSeconds(), completedThisRun = 0

            func report(_ phase: TrainingPhase, _ message: String) {
                progress.phase = phase; progress.message = message; progress.step = step
                progress.epoch = epoch; progress.cursor = cursor; progress.stepsPerEpoch = schedule.count
                progress.loss = step > 0 ? lastLoss : nil; progress.validationLoss = validationLoss
                progress.activeMemory = Memory.activeMemory; progress.cacheMemory = Memory.cacheMemory
                progress.residentMemory = residentBytes(); progress.elapsedSeconds = Date().timeIntervalSince(started)
                let now = Date(), cpu = processCPUSeconds(), interval = now.timeIntervalSince(sampleTime)
                if interval > 0.25 { progress.cpuPercent = max(0, (cpu - sampleCPU) / interval * 100); sampleCPU = cpu; sampleTime = now }
                progress.stepsPerSecond = Double(completedThisRun) / max(0.001, progress.elapsedSeconds)
                publish(progress)
            }

            func saveCheckpoint(isBest: Bool) throws {
                guard step > 0 else { return }
                report(.checkpointing, "Writing weights, optimizer and temporal state…")
                let manifest = CheckpointManifest(modelID: request.model.id, configuration: request.model.configuration,
                    configurationFingerprint: request.model.configuration.fingerprint, datasetFingerprint: dataset.fingerprint,
                    trainingRecordingIDs: dataset.training.map { $0.item.id }, validationRecordingIDs: dataset.validation.map { $0.item.id },
                    stage: request.stage, settings: request.settings, step: step, epoch: epoch, sampleCursor: cursor,
                    trainingLoss: lastLoss, validationLoss: validationLoss, bestValidationLoss: bestLoss)
                var evaluatedManifest = manifest
                evaluatedManifest.actionEvaluation = progress.actionEvaluation
                let saved = try checkpoints.save(evaluatedManifest, isBest: isBest) { directory in
                    try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: directory.appendingPathComponent("weights.safetensors"))
                    var state = optimizer.arrays()
                    for (index, value) in hidden.enumerated() { state["carry.\(index)"] = value }
                    try MLX.save(arrays: state, url: directory.appendingPathComponent("optimizer.safetensors"))
                }
                progress.checkpoint = saved.id; checkpointSaved(saved)
            }

            while epoch < request.settings.epochs {
                if control.current != .run || Date() >= deadline {
                    try saveCheckpoint(isBest: false)
                    let cancelled = control.current == .cancel
                    report(cancelled ? .cancelled : .paused, cancelled ? "Cancelled safely. The latest checkpoint is preserved." : "Paused with a resumable checkpoint.")
                    return
                }
                guard let plan = schedule.plan(at: cursor) else {
                    let validationSchedule = SequenceSchedule(recordings: dataset.validation, batchSize: request.settings.batchSize,
                        sequenceLength: model.configuration.sequenceLength, seed: request.settings.seed)
                    var weightedLoss: Double = 0, examples = 0, validationHidden: [MLXArray] = []
                    var evaluation = ActionEvaluation()
                    model.train(false)
                    for index in 0..<validationSchedule.count {
                        if control.current != .run || Date() >= deadline { break }
                        report(.validating, "Evaluating held-out recordings · \(index + 1)/\(validationSchedule.count)")
                        guard let plan = validationSchedule.plan(at: index) else { break }
                        if plan.resetsMemory { validationHidden = [] }
                        try autoreleasepool {
                            let batch = try TrainingBatch.load(plan: plan, configuration: model.configuration, checkCancellation: {})
                            let output = PolicyLoss.forward(model, batch.arrays + validationHidden)
                            let loss = PolicyLoss.loss(output, batch.arrays, stage: stage)
                            validationHidden = output.hidden.map { stopGradient($0) }
                            eval(loss, validationHidden)
                            let value = loss.item(Float.self)
                            guard value.isFinite else { throw DataIntegrityError.invalidData("Validation loss became non-finite. The previous checkpoint remains available.") }
                            weightedLoss += Double(value) * Double(batch.validCount); examples += batch.validCount
                            if stage == .imitation {
                                evaluation.add(ActionEvaluation.measure(logits: output.actionLogits,
                                    targets: batch.arrays[BatchField.actions.rawValue], valid: batch.arrays[BatchField.valid.rawValue],
                                    mask: batch.arrays[BatchField.actionMask.rawValue]))
                            }
                        }
                    }
                    model.train(true)
                    // Only a complete held-out pass can select the best checkpoint.
                    let validationComplete = examples > 0 && control.current == .run && Date() < deadline
                    validationLoss = validationComplete ? Float(weightedLoss / Double(examples)) : nil
                    progress.actionEvaluation = validationComplete && stage == .imitation ? evaluation : nil
                    let improved = validationLoss.map { $0 < (bestLoss ?? .infinity) } ?? false
                    if improved { bestLoss = validationLoss }
                    progress.history.append(LossPoint(step: step, training: lastLoss, validation: validationLoss))
                    epoch += 1; cursor = 0; hidden = []
                    try saveCheckpoint(isBest: improved)
                    schedule = SequenceSchedule(recordings: dataset.training, batchSize: request.settings.batchSize,
                        sequenceLength: model.configuration.sequenceLength, seed: request.settings.seed &+ UInt64(epoch))
                    continue
                }
                if plan.resetsMemory { hidden = [] }
                report(.training, stage == .pretraining ? "Learning action-conditioned visual dynamics" : "Learning demonstrated actions and timing")
                try autoreleasepool {
                    let batch = try TrainingBatch.load(plan: plan, configuration: model.configuration, checkCancellation: {})
                    let (values, gradients) = lossGradient(model, batch.arrays + hidden)
                    let loss = values[0].item(Float.self)
                    guard loss.isFinite else { throw DataIntegrityError.invalidData("Training loss became non-finite. Reduce the learning rate and start from a valid checkpoint.") }
                    let norm = optimizer.update(model: model, gradients: gradients, clip: request.settings.gradientClip).item(Float.self)
                    guard norm.isFinite else { throw DataIntegrityError.invalidData("Gradients became non-finite. The previous checkpoint remains unchanged.") }
                    hidden = Array(values.dropFirst()); eval(hidden)
                    lastLoss = loss; progress.gradientNorm = norm
                }
                step += 1; cursor += 1; completedThisRun += 1
                if step % 5 == 0 || step == 1 {
                    progress.history.append(LossPoint(step: step, training: lastLoss))
                    if progress.history.count > 500 { progress.history.removeFirst(progress.history.count - 500) }
                }
                if step % request.settings.checkpointInterval == 0 { try saveCheckpoint(isBest: false) }
                report(.training, "Training locally on Apple Silicon")
            }
            report(.complete, "All epochs completed. \(dataset.validation.isEmpty ? "No held-out recordings were available; this is training loss only." : "The latest and best validation checkpoints are preserved.")")
        } catch is CancellationError {
            progress.phase = control.current == .cancel ? .cancelled : .paused
            progress.message = "Stopped during preparation. No model weights were changed."; publish(progress)
        } catch {
            progress.phase = .failed; progress.message = error.localizedDescription; publish(progress)
        }
    }

    static func restoreWeights(_ model: PolicyNetwork, from directory: URL) throws {
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        guard Set(expected.keys) == Set(weights.keys), expected.allSatisfy({ weights[$0.key]?.shape == $0.value.shape }) else {
            throw DataIntegrityError.invalidData("Checkpoint tensor names or shapes do not match the model.")
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
    }

    private static func processCPUSeconds() -> Double {
        var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
        return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6
    }
    private static func residentBytes() -> UInt64 {
        var info = mach_task_basic_info(), count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}

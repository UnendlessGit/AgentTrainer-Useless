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
    var trainingNonWaitExamples = 0
    var validationRecordings = 0
    var excludedOutsideTarget = 0
    var loss: Float?
    var validationLoss: Float?
    var actionEvaluation: ActionEvaluation?
    var evaluationEpoch: Int?
    var trainingPointerExamples = 0
    var gradientNorm: Float?
    var learningRate: Float = 0
    var stepsPerSecond = 0.0
    var activeMemory = 0
    var cacheMemory = 0
    var physicalFootprint: UInt64?
    var cpuPercent = 0.0
    var elapsedSeconds = 0.0
    var checkpoint: UUID?
    var initialWeights: InitialWeights?
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
            try request.preferences.validate()
            try request.settings.validate()
            try request.model.configuration.validate()
            if !request.resume && request.stage == .imitation && request.settings.startingWeights == .trained && !request.model.canRun {
                throw DataIntegrityError.invalidData("This model has no compatible trained checkpoint. Choose pre-trained or new weights, or restore the trained configuration.")
            }
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
            progress.trainingNonWaitExamples = dataset.nonWaitExampleCount
            progress.validationRecordings = dataset.validation.count
            progress.excludedOutsideTarget = dataset.excludedOutsideTarget
            let pointerToken = PolicyActionCodec(capabilities: request.model.configuration.capabilities).token(for: .pointer(x: 0, y: 0))
            progress.trainingPointerExamples = pointerToken.map { token in dataset.stateFrequencies.reduce(0) { $0 + ($1.actionCounts[token] ?? 0) } } ?? 0
            MLXRandom.seed(request.settings.seed)
            let model = PolicyNetwork(configuration: request.model.configuration)
            let optimizer = ResumableAdamW(learningRate: request.settings.learningRate, weightDecay: request.settings.weightDecay)
            let checkpoints = CheckpointStore(root: URL(fileURLWithPath: request.preferences.checkpointsPath))
            var epoch = 0, cursor = 0, step = 0
            var hidden: [MLXArray] = [], lastLoss: Float = 0, validationLoss: Float?, bestLoss: Float?
            var bestHasMatchedControl = false
            var carries: [Int: [MLXArray]] = [:], restoredState: [String: MLXArray]?
            var initialWeights: InitialWeights? = .scratch
            var trainingRunID: UUID? = UUID()
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
                if request.settings.interleavesRecordings { restoredState = state }
                else if saved.sampleCursor > 0 {
                    let count = model.configuration.memory == .recurrent ? model.configuration.memoryDepth : 1
                    hidden = try (0..<count).map { index in
                        guard let value = state["carry.\(index)"] else { throw DataIntegrityError.invalidData("This checkpoint is missing temporal state.") }
                        return value
                    }
                }
                epoch = saved.epoch; cursor = saved.sampleCursor; step = saved.step
                lastLoss = saved.trainingLoss; validationLoss = saved.validationLoss; bestLoss = saved.bestValidationLoss
                bestHasMatchedControl = saved.bestHasMatchedControl ?? false
                progress.actionEvaluation = saved.actionEvaluation
                progress.evaluationEpoch = saved.actionEvaluation == nil ? nil : saved.epoch
                progress.checkpoint = saved.id
                initialWeights = saved.initialWeights
                trainingRunID = saved.trainingRunID
                guard epoch < request.settings.epochs else { throw DataIntegrityError.invalidData("This run already completed all epochs. Start a new run to train again.") }
            } else if request.stage == .imitation {
                let source: (String?, TrainingStage)?
                switch request.settings.startingWeights {
                case .automatic:
                    source = request.model.pretrainingCompatible ? (request.model.pretrainedCheckpoint, .pretraining) : nil
                case .trained: source = (request.model.trainedCheckpoint, .imitation)
                case .scratch: source = nil
                }
                if let (identifier, stage) = source {
                    guard let identifier, let id = UUID(uuidString: identifier) else {
                        throw DataIntegrityError.invalidData("The starting checkpoint identifier is invalid.")
                    }
                    let (saved, directory) = try checkpoints.load(modelID: request.model.id, checkpointID: id, configuration: request.model.configuration)
                    guard saved.stage == stage else { throw DataIntegrityError.invalidData("The starting checkpoint has the wrong training stage.") }
                    try restoreWeights(model, from: directory)
                    initialWeights = .checkpoint(id: id, stage: stage)
                }
            }
            progress.initialWeights = initialWeights
            eval(model)
            model.train(true)
            let stage = request.stage
            let capabilities = stage == .imitation ? request.settings.imitationCapabilities(model.configuration.capabilities)
                : model.configuration.capabilities
            let cursorIndependent = stage == .imitation && request.settings.usesCursorIndependentKeys
            let actionBalance = request.settings.balancesActionFrequency
                ? ((request.settings.actionBalanceVersion ?? 1) >= 2 ? PolicyLoss.ActionBalance(frequencies: dataset.stateFrequencies,
                    balanceTokens: request.settings.actionBalanceVersion == 3,
                    recordings: request.settings.actionBalanceVersion == 3 ? dataset.training : [])
                    : PolicyLoss.ActionBalance(total: dataset.exampleCount, inputs: dataset.nonWaitExampleCount)) : nil
            let lossGradient = valueAndGrad(model: model) { model, arrays in
                PolicyLoss.values(model, arrays, stage: stage, balanceInputChoices: request.settings.balancesInputChoices,
                                  actionBalance: actionBalance, wholeSceneDetail: cursorIndependent)
            }
            var schedule = SequenceSchedule(recordings: dataset.training, batchSize: request.settings.batchSize,
                sequenceLength: model.configuration.sequenceLength, seed: request.settings.seed &+ UInt64(epoch),
                interleaved: request.settings.interleavesRecordings, rotated: request.settings.rotatesSequences)
            guard (0...schedule.count).contains(cursor), epoch >= 0, optimizer.step == step else {
                throw DataIntegrityError.invalidData("The checkpoint cursor or optimizer step is inconsistent.")
            }
            if let state = restoredState {
                let c = model.configuration, count = c.memory == .recurrent ? c.memoryDepth : 1
                var expected: Set<String> = []
                for group in schedule.pendingMemoryGroups(at: cursor) {
                    let shape = c.memory == .recurrent ? [schedule.groups[group].count, c.memorySize]
                        : [schedule.groups[group].count, c.sequenceLength - 1, c.memorySize]
                    carries[group] = try (0..<count).map { index in
                        let key = "carry.\(group).\(index)"
                        expected.insert(key)
                        guard let value = state[key], value.shape == shape else {
                            throw DataIntegrityError.invalidData("This checkpoint is missing compatible temporal state for a recording group.")
                        }
                        return value
                    }
                }
                guard Set(state.keys.filter { $0.hasPrefix("carry.") }) == expected else {
                    throw DataIntegrityError.invalidData("The checkpoint's carried recording groups do not match its schedule.")
                }
                restoredState = nil
            } else if !request.settings.interleavesRecordings, let plan = schedule.plan(at: cursor), !plan.resetsMemory {
                let c = model.configuration
                let count = c.memory == .recurrent ? c.memoryDepth : 1
                let shape = c.memory == .recurrent ? [plan.recordings.count, c.memorySize]
                    : [plan.recordings.count, c.sequenceLength - 1, c.memorySize]
                guard hidden.count == count, hidden.allSatisfy({ $0.shape == shape }) else {
                    throw DataIntegrityError.invalidData("The checkpoint's temporal state has incompatible dimensions.")
                }
            }
            var sampleTime = Date(), sampleCPU = processCPUSeconds(), completedThisRun = 0

            func report(_ phase: TrainingPhase, _ message: String) {
                progress.phase = phase; progress.message = message; progress.step = step
                progress.epoch = epoch; progress.cursor = cursor; progress.stepsPerEpoch = schedule.count
                progress.loss = step > 0 ? lastLoss : nil; progress.validationLoss = validationLoss
                progress.activeMemory = Memory.activeMemory; progress.cacheMemory = Memory.cacheMemory
                progress.physicalFootprint = footprintBytes(); progress.elapsedSeconds = Date().timeIntervalSince(started)
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
                    trainingLoss: lastLoss, validationLoss: cursor == 0 ? validationLoss : nil, bestValidationLoss: bestLoss)
                var evaluatedManifest = manifest
                evaluatedManifest.initialWeights = initialWeights
                evaluatedManifest.trainingRunID = trainingRunID
                evaluatedManifest.bestHasMatchedControl = bestHasMatchedControl
                // Mid-epoch weights have changed since the last validation pass.
                // Its scores remain useful in the live UI, but do not describe
                // these saved weights. Only epoch-boundary saves own the scores.
                evaluatedManifest.actionEvaluation = cursor == 0 ? progress.actionEvaluation : nil
                let saved = try autoreleasepool {
                    try checkpoints.save(evaluatedManifest, isBest: isBest) { directory in
                        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: directory.appendingPathComponent("weights.safetensors"))
                        var state = optimizer.arrays()
                        if request.settings.interleavesRecordings {
                            for (group, values) in carries {
                                for (index, value) in values.enumerated() { state["carry.\(group).\(index)"] = value }
                            }
                        } else {
                            for (index, value) in hidden.enumerated() { state["carry.\(index)"] = value }
                        }
                        try MLX.save(arrays: state, url: directory.appendingPathComponent("optimizer.safetensors"))
                    }
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
                            let batch = try TrainingBatch.load(plan: plan, configuration: model.configuration, stage: stage,
                                actionBalance: actionBalance, capabilities: capabilities, cursorIndependent: cursorIndependent, checkCancellation: {})
                            let output = PolicyLoss.forward(model, batch.arrays + validationHidden, wholeSceneDetail: cursorIndependent)
                            let loss = PolicyLoss.loss(output, batch.arrays, stage: stage, balanceInputChoices: request.settings.balancesInputChoices,
                                                       actionBalance: actionBalance)
                            validationHidden = output.hidden.map { stopGradient($0) }
                            eval(loss, validationHidden)
                            let value = loss.item(Float.self)
                            guard value.isFinite else { throw DataIntegrityError.invalidData("Validation loss became non-finite. The previous checkpoint remains available.") }
                            weightedLoss += Double(value) * Double(batch.validCount); examples += batch.validCount
                            if stage == .imitation {
                                evaluation.add(ActionEvaluation.measure(logits: output.actionLogits,
                                    targets: batch.arrays[BatchField.actions.rawValue], valid: batch.arrays[BatchField.valid.rawValue],
                                    mask: batch.arrays[BatchField.actionMask.rawValue],
                                    actions: PolicyActionCodec(capabilities: model.configuration.capabilities).actions,
                                    hierarchical: request.settings.balancesActionFrequency,
                                    context: batch.arrays[BatchField.context.rawValue]))
                            }
                        }
                    }
                    model.train(true)
                    // Keep the cursor at the end of training until validation
                    // finishes. Resume then repeats this held-out pass using the
                    // same weights instead of silently skipping it (or claiming
                    // completion when interrupted in the final epoch).
                    if control.current != .run || Date() >= deadline {
                        validationLoss = nil; progress.actionEvaluation = nil; progress.evaluationEpoch = nil
                        try saveCheckpoint(isBest: false)
                        let cancelled = control.current == .cancel
                        report(cancelled ? .cancelled : .paused,
                               cancelled ? "Cancelled safely during validation. Resume will repeat the held-out pass."
                               : "Paused during validation. Resume will repeat the held-out pass.")
                        return
                    }
                    // Only a complete held-out pass can select the best checkpoint.
                    let validationComplete = examples > 0
                    validationLoss = validationComplete ? Float(weightedLoss / Double(examples)) : nil
                    progress.actionEvaluation = validationComplete && stage == .imitation ? evaluation : nil
                    progress.evaluationEpoch = progress.actionEvaluation == nil ? nil : epoch + 1
                    let matchedControl = progress.actionEvaluation?.hasMatchedControl ?? false
                    let improved = validationLoss.map {
                        prefersCheckpoint(loss: $0, matchedControl: matchedControl, bestLoss: bestLoss,
                            bestHasMatchedControl: bestHasMatchedControl,
                            requireMatchedControl: stage == .imitation && request.settings.prefersMatchedControlCheckpoints)
                    } ?? false
                    if improved { bestLoss = validationLoss; bestHasMatchedControl = matchedControl }
                    progress.history.append(LossPoint(step: step, training: lastLoss, validation: validationLoss))
                    epoch += 1; cursor = 0; hidden = []; carries = [:]
                    try saveCheckpoint(isBest: improved)
                    schedule = SequenceSchedule(recordings: dataset.training, batchSize: request.settings.batchSize,
                        sequenceLength: model.configuration.sequenceLength, seed: request.settings.seed &+ UInt64(epoch),
                        interleaved: request.settings.interleavesRecordings, rotated: request.settings.rotatesSequences)
                    continue
                }
                if plan.resetsMemory { hidden = [] }
                var batchHidden = request.settings.interleavesRecordings ? (plan.resetsMemory ? [] : (carries[plan.memoryGroup] ?? [])) : hidden
                if plan.warmupChunks > 0 {
                    // Warm the exact causal prefix with current weights. Random
                    // epoch offsets must never inject future memory into earlier
                    // frames, or approximate a long memory with just one chunk.
                    var warmed: [MLXArray] = [], interrupted = false
                    for chunk in 0..<plan.warmupChunks {
                        if control.current != .run || Date() >= deadline { interrupted = true; break }
                        report(.training, "Warming recording history · \(chunk + 1)/\(plan.warmupChunks)")
                        try autoreleasepool {
                            let prefix = try TrainingBatch.load(plan: SequenceBatchPlan(recordings: plan.recordings, chunk: chunk),
                                configuration: model.configuration, stage: stage, actionBalance: actionBalance, capabilities: capabilities,
                                cursorIndependent: cursorIndependent, checkCancellation: {})
                            let output = PolicyLoss.forward(model, prefix.arrays + warmed, wholeSceneDetail: cursorIndependent)
                            warmed = output.hidden.map { stopGradient($0) }; eval(warmed)
                        }
                    }
                    if interrupted || control.current != .run || Date() >= deadline {
                        try saveCheckpoint(isBest: false)
                        let cancelled = control.current == .cancel
                        report(cancelled ? .cancelled : .paused,
                               step == 0 ? "Stopped before the first optimizer update. No new checkpoint was created; start a new run to try again."
                               : cancelled ? "Cancelled safely during memory warm-up." : "Paused during memory warm-up. Resume will repeat the causal prefix.")
                        return
                    }
                    batchHidden = warmed
                }
                report(.training, stage == .pretraining ? "Learning action-conditioned visual dynamics" : "Learning demonstrated actions and timing")
                try autoreleasepool {
                    let batch = try TrainingBatch.load(plan: plan, configuration: model.configuration, stage: stage,
                        actionBalance: actionBalance, capabilities: capabilities, cursorIndependent: cursorIndependent, checkCancellation: {})
                    let (values, gradients) = lossGradient(model, batch.arrays + batchHidden)
                    let loss = values[0].item(Float.self)
                    guard loss.isFinite else { throw DataIntegrityError.invalidData("Training loss became non-finite. Reduce the learning rate and start from a valid checkpoint.") }
                    let clip = request.settings.gradientClip * (request.settings.balancesGradientClipping ? batch.gradientWeightScale : 1)
                    let norm = optimizer.update(model: model, gradients: gradients, clip: clip).item(Float.self)
                    guard norm.isFinite else { throw DataIntegrityError.invalidData("Gradients became non-finite. The previous checkpoint remains unchanged.") }
                    let nextHidden = Array(values.dropFirst()); eval(nextHidden)
                    if request.settings.interleavesRecordings {
                        carries[plan.memoryGroup] = plan.endsMemory ? nil : nextHidden
                    } else { hidden = nextHidden }
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

    static func prefersCheckpoint(loss: Float, matchedControl: Bool, bestLoss: Float?, bestHasMatchedControl: Bool,
                                  requireMatchedControl: Bool) -> Bool {
        if requireMatchedControl, bestLoss != nil, matchedControl != bestHasMatchedControl { return matchedControl }
        return loss < (bestLoss ?? .infinity)
    }

    private static func processCPUSeconds() -> Double {
        var usage = rusage(); getrusage(RUSAGE_SELF, &usage)
        return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6
    }
    private static func footprintBytes() -> UInt64? {
        var info = task_vm_info_data_t(), count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : nil
    }
}

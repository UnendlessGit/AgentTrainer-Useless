import Foundation
import MLX
import MLXRandom

struct RunRequest: Sendable {
    var model: AIModel
    var configuration: RunConfiguration
    var preferences: AppPreferences
    var target: RunTargetGuard
}

enum RunWorker {
    static func run(_ request: RunRequest, source: CaptureFrameSource, clock: SessionClock, executor: AgentInputExecutor,
                    humanState: @escaping @Sendable () -> InputState,
                    publish: @escaping @Sendable (RunProgress, CapturedScene?) -> Void,
                    finished: @escaping @Sendable (String) -> Void) {
        var progress = RunProgress()
        var started: UInt64?
        defer {
            if let started { progress.elapsed = Double(clock.now - min(clock.now, started)) / 1e9 }
            progress.heldInput = executor.state
            publish(progress, nil)
            Memory.clearCache(); finished(executor.stopReason ?? "Run stopped.")
        }
        do {
            let configuration = request.model.configuration
            try configuration.validate()
            let checkpoints = CheckpointStore(root: URL(fileURLWithPath: request.preferences.checkpointsPath))
            guard let identifier = request.model.trainedCheckpoint.flatMap(UUID.init(uuidString:)) else {
                throw DataIntegrityError.invalidData("Choose a model with a compatible imitation-learning checkpoint.")
            }
            let (_, directory) = try checkpoints.inference(modelID: request.model.id, latestID: identifier,
                configuration: configuration, preferBest: request.configuration.useBestCheckpoint)
            Memory.memoryLimit = request.preferences.memoryLimitGB * 1_073_741_824
            Memory.cacheLimit = request.preferences.cacheLimitGB * 1_073_741_824
            let model = PolicyNetwork(configuration: configuration)
            try TrainingWorker.restoreWeights(model, from: directory)
            model.train(false); eval(model)
            MLXRandom.seed(42)
            let permissions = configuration.capabilities.intersecting(request.configuration.permissions)
            let runner = PolicyRunner(model: model, permissions: permissions, instruction: request.configuration.instruction)
            let start = clock.now, deadline = start + UInt64(request.configuration.maximumRunSeconds) * 1_000_000_000
            started = start
            var previous: ComputerAction = .wait(seconds: 0), previousTime = start, lastPublish = start
            while !executor.isStopped && clock.now < deadline {
                try autoreleasepool {
                    let human = humanState()
                    if !human.keys.isEmpty || !human.buttons.isEmpty {
                        if request.configuration.stopOnHumanInput { executor.stop("Stopped by keyboard or mouse input."); return }
                        // In sharing mode, human-held modifiers/buttons must never
                        // combine with a new agent press. Resume after release.
                        progress.waitingForHuman = true
                        if clock.now >= lastPublish + 250_000_000 {
                            progress.elapsed = Double(clock.now - start) / 1e9
                            progress.heldInput = executor.state
                            publish(progress, nil); lastPublish = clock.now
                        }
                        Thread.sleep(forTimeInterval: 0.01); return
                    }
                    progress.waitingForHuman = false
                    guard let scene = try source.latestScene() else { Thread.sleep(forTimeInterval: 0.01); return }
                    try request.target.validate(observedBounds: scene.bounds)
                    let decisionTime = clock.now
                    let decision = try runner.decide(scene: scene, state: executor.state, previousAction: previous,
                        elapsed: Double(decisionTime - min(decisionTime, previousTime)) / 1e9,
                        sourceAge: Double(decisionTime - min(decisionTime, scene.sourceTime)) / 1e9,
                        deterministic: request.configuration.deterministic, temperature: request.configuration.temperature)
                    let scheduled = decisionTime + UInt64(decision.delay * 1e9)
                    while clock.now < scheduled && !executor.isStopped && clock.now < deadline {
                        Thread.sleep(forTimeInterval: min(0.005, Double(scheduled - min(scheduled, clock.now)) / 1e9))
                    }
                    guard !executor.isStopped, clock.now < deadline else { return }
                    let beforeEmission = humanState()
                    guard beforeEmission.keys.isEmpty, beforeEmission.buttons.isEmpty else {
                        throw DataIntegrityError.invalidData("Human input began during a decision. The run stopped before emitting that action.")
                    }
                    try request.target.validate(observedBounds: decision.observationBounds, action: decision.action, state: executor.state)
                    try executor.execute(decision.action, bounds: decision.observationBounds)
                    previous = decision.action; previousTime = clock.now
                    progress.record(decision.action); progress.elapsed = Double(clock.now - start) / 1e9
                    progress.inferenceMilliseconds = decision.inferenceSeconds * 1000
                    progress.lastAction = decision.action.label
                    progress.history.append(String(format: "%.2fs  %@", progress.elapsed, decision.action.label))
                    if progress.history.count > 80 { progress.history.removeFirst(progress.history.count - 80) }
                    if clock.now >= lastPublish + 250_000_000 {
                        progress.activeMemory = Memory.activeMemory; progress.cacheMemory = Memory.cacheMemory
                        progress.heldInput = executor.state
                        publish(progress, scene); lastPublish = clock.now
                    }
                }
            }
            executor.stop("Run time limit reached.")
        } catch is CancellationError { executor.stop("Run stopped.") }
        catch { executor.stop(error.localizedDescription) }
    }
}

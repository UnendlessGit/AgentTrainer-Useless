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
                    publish: @escaping @Sendable (RunProgress, CapturedScene?) -> Void,
                    finished: @escaping @Sendable (String) -> Void) {
        var progress = RunProgress()
        defer { Memory.clearCache(); finished(executor.stopReason ?? "Run stopped.") }
        do {
            let configuration = request.model.configuration
            try configuration.validate()
            let checkpoints = CheckpointStore(root: URL(fileURLWithPath: request.preferences.checkpointsPath))
            var identifier = request.model.trainedCheckpoint.flatMap(UUID.init(uuidString:))
            if request.configuration.useBestCheckpoint {
                let pointerURL = checkpoints.root.appendingPathComponent(request.model.id.uuidString).appendingPathComponent("best-imitation.json")
                if FileManager.default.fileExists(atPath: pointerURL.path) {
                    identifier = try AtomicFile.decode(CheckpointPointer.self, from: pointerURL).checkpointID
                }
            }
            guard let identifier else { throw DataIntegrityError.invalidData("Choose a model with a compatible imitation-learning checkpoint.") }
            let (metadata, directory) = try checkpoints.load(modelID: request.model.id, checkpointID: identifier, configuration: configuration)
            guard metadata.stage == .imitation else { throw DataIntegrityError.invalidData("Only imitation-learning checkpoints can control input.") }
            Memory.memoryLimit = request.preferences.memoryLimitGB * 1_073_741_824
            Memory.cacheLimit = request.preferences.cacheLimitGB * 1_073_741_824
            let model = PolicyNetwork(configuration: configuration)
            try TrainingWorker.restoreWeights(model, from: directory)
            model.train(false); eval(model)
            MLXRandom.seed(42)
            let permissions = configuration.capabilities.intersecting(request.configuration.permissions)
            let runner = PolicyRunner(model: model, permissions: permissions, instruction: request.configuration.instruction)
            let start = clock.now, deadline = start + UInt64(request.configuration.maximumRunSeconds) * 1_000_000_000
            var previous: ComputerAction = .wait(seconds: 0), previousTime = start, lastPublish = start
            while !executor.isStopped && clock.now < deadline {
                try autoreleasepool {
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
                    try request.target.validate(observedBounds: decision.observationBounds, action: decision.action, state: executor.state)
                    try executor.execute(decision.action, bounds: decision.observationBounds)
                    previous = decision.action; previousTime = clock.now
                    progress.decisions += 1; progress.elapsed = Double(clock.now - start) / 1e9
                    progress.inferenceMilliseconds = decision.inferenceSeconds * 1000
                    progress.lastAction = decision.action.label
                    progress.history.append(String(format: "%.2fs  %@", progress.elapsed, decision.action.label))
                    if progress.history.count > 80 { progress.history.removeFirst(progress.history.count - 80) }
                    if clock.now >= lastPublish + 250_000_000 {
                        progress.activeMemory = Memory.activeMemory; progress.cacheMemory = Memory.cacheMemory
                        publish(progress, scene); lastPublish = clock.now
                    }
                }
            }
            executor.stop("Run time limit reached.")
            publish(progress, nil)
        } catch is CancellationError { executor.stop("Run stopped.") }
        catch { executor.stop(error.localizedDescription) }
    }
}

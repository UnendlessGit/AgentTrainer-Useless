import XCTest
import MLX
import MLXNN
import ImageIO
import UniformTypeIdentifiers
@testable import AgentTrainer

private final class ProgressCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var updates: [TrainingProgress] = []
    func append(_ progress: TrainingProgress) { lock.withLock { updates.append(progress) } }
    var last: TrainingProgress? { lock.withLock { updates.last } }
}

final class TrainingPipelineTests: XCTestCase {
    private func configuration() -> PolicyConfiguration {
        var c = PolicyConfiguration()
        c.imageSize = 64; c.visualWidth = 32; c.visualDepth = 1; c.memorySize = 32; c.memoryDepth = 1
        c.sequenceLength = 2; c.detailCrop = false; c.instructionConditioning = false
        c.capabilities.keys = [0]
        return c
    }

    private func recording(root: URL, kind: LibraryKind = .imitation) throws -> RecordingItem {
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Recorded fixture", folderID: UUID(), kind: kind,
            target: CaptureTarget(), settings: RecordingSettings()))
        for index in 0..<6 {
            let data = Data([UInt8(index * 40), 30, 180, 255])
            let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
            let image = try XCTUnwrap(CGImage(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let file = "frames/\(index).jpg", url = journal.url.appendingPathComponent(file)
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            let time = UInt64(index + 1) * 100_000_000
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time, sourceTimeNanoseconds: time,
                imageFile: file, width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)), state: InputState(), reusedPixels: false))
            if index < 5 {
                try journal.append(event: InputTransition(id: 0, timeNanoseconds: time + 10_000_000, action: .keyDown(code: 0)))
                try journal.append(event: InputTransition(id: 0, timeNanoseconds: time + 20_000_000, action: .keyUp(code: 0)))
            }
        }
        try journal.finish(at: 650_000_000)
        return RecordingItem(manifest: journal.snapshot, edits: RecordingEdits(), url: journal.url)
    }

    func testDiskDatasetSeparatesRecordingsAndRejectsUnsupportedActions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try [recording(root: root), recording(root: root), recording(root: root)]
        let c = configuration(), settings = TrainingSettings()
        let dataset = try PreparedDataset.prepare(items: items, configuration: c, settings: settings, stage: .imitation,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        XCTAssertEqual(dataset.validation.count, 1); XCTAssertEqual(dataset.training.count, 2)
        XCTAssertTrue(Set(dataset.training.map { $0.item.id }).isDisjoint(with: dataset.validation.map { $0.item.id }))
        let schedule = SequenceSchedule(recordings: dataset.training, batchSize: 2, sequenceLength: 2, seed: 42)
        let plan = try XCTUnwrap(schedule.plan(at: 1))
        XCTAssertFalse(plan.resetsMemory)
        for lane in plan.recordings {
            let examples = try lane.examples(start: 2, count: 2)
            XCTAssertEqual(examples.count, 2)
            XCTAssertTrue(examples.allSatisfy { $0.recordingID == lane.item.id })
            XCTAssertEqual(examples[0].targetAction, .keyDown(code: 0))
        }
        var incompatible = c; incompatible.capabilities.keys = []
        XCTAssertThrowsError(try PreparedDataset.prepare(items: items, configuration: incompatible, settings: settings, stage: .imitation,
            root: root.appendingPathComponent("invalid"), checkCancellation: {}, progress: { _ in }))
    }

    func testOmittingFutureImageWorkPreservesImitationLossAndGradients() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root), c = configuration()
        let dataset = try PreparedDataset.prepare(items: [item], configuration: c, settings: TrainingSettings(), stage: .imitation,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        let plan = SequenceBatchPlan(recordings: dataset.training, chunk: 0)
        let full = try TrainingBatch.load(plan: plan, configuration: c, stage: .pretraining, checkCancellation: {})
        let imitation = try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation, checkCancellation: {})
        let model = PolicyNetwork(configuration: c)
        let lossGradient = valueAndGrad(model: model) { model, arrays in PolicyLoss.values(model, arrays, stage: .imitation) }
        let (fullLoss, fullGradients) = lossGradient(model, full.arrays)
        let (imitationLoss, imitationGradients) = lossGradient(model, imitation.arrays)
        XCTAssertEqual(fullLoss[0].item(Float.self), imitationLoss[0].item(Float.self), accuracy: 1e-6)
        let expected = Dictionary(uniqueKeysWithValues: fullGradients.flattened())
        for (name, gradient) in imitationGradients.flattened() {
            XCTAssertLessThanOrEqual(max(abs(gradient - expected[name]!)).item(Float.self), 1e-6, name)
        }
        XCTAssertGreaterThan(sum(abs(full.arrays[BatchField.futurePixels.rawValue])).item(Float.self), 0)
        XCTAssertEqual(sum(abs(imitation.arrays[BatchField.futurePixels.rawValue])).item(Float.self), 0)
    }

    func testWorkerPauseResumeMatchesUninterruptedTrainingAndSavesValidation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try [recording(root: root), recording(root: root)]
        var model = AIModel(name: "Training pipeline"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1; settings.checkpointInterval = 100
        var preferences = AppPreferences.defaults(at: root)
        preferences.memoryLimitGB = 4; preferences.cacheLimitGB = 1
        let pause = TrainingControl(), first = ProgressCollector()
        let request = TrainingRequest(model: model, settings: settings, stage: .imitation, items: items, preferences: preferences, resume: false)
        TrainingWorker.run(request, control: pause, publish: { update in
            first.append(update)
            if update.phase == .training && update.step == 1 { pause.set(.pause) }
        }, checkpointSaved: { _ in })
        XCTAssertEqual(first.last?.phase, .paused, first.last?.message ?? "Missing progress")
        XCTAssertNotNil(first.last?.checkpoint)
        let resumed = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items, preferences: preferences, resume: true),
            control: TrainingControl(), publish: resumed.append, checkpointSaved: { _ in })
        XCTAssertEqual(resumed.last?.phase, .complete, resumed.last?.message ?? "Missing progress")
        XCTAssertNotNil(resumed.last?.validationLoss)
        XCTAssertEqual(resumed.last?.actionEvaluation?.total, 11)
        XCTAssertEqual(resumed.last?.actionEvaluation?.nonWaitTotal, 10)
        let checkpoints = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
        let resumedCheckpoint = try XCTUnwrap(checkpoints.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        let resumedWeights = try MLX.loadArrays(url: resumedCheckpoint.1.appendingPathComponent("weights.safetensors"))
        let uninterrupted = ProgressCollector()
        TrainingWorker.run(request, control: TrainingControl(), publish: uninterrupted.append, checkpointSaved: { _ in })
        XCTAssertEqual(uninterrupted.last?.phase, .complete, uninterrupted.last?.message ?? "Missing progress")
        let fullCheckpoint = try XCTUnwrap(checkpoints.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        let fullWeights = try MLX.loadArrays(url: fullCheckpoint.1.appendingPathComponent("weights.safetensors"))
        for key in resumedWeights.keys {
            XCTAssertLessThan(max(abs(resumedWeights[key]! - fullWeights[key]!)).item(Float.self), 1e-5, key)
        }
        XCTAssertEqual(resumedCheckpoint.0.step, fullCheckpoint.0.step)
        XCTAssertEqual(resumedCheckpoint.0.validationRecordingIDs, fullCheckpoint.0.validationRecordingIDs)
    }

    func testPretrainingWorkerProducesFiniteCheckpointAndImitationContinuesFromIt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root, kind: .pretraining)
        var model = AIModel(name: "Pretraining pipeline"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1
        let preferences = AppPreferences.defaults(at: root), progress = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .pretraining, items: [item], preferences: preferences, resume: false),
            control: TrainingControl(), publish: progress.append, checkpointSaved: { _ in })
        XCTAssertEqual(progress.last?.phase, .complete, progress.last?.message ?? "Missing progress")
        let checkpoint = try XCTUnwrap(progress.last?.checkpoint)
        model.pretrainedCheckpoint = checkpoint.uuidString
        model.pretrainedFingerprint = model.configuration.fingerprint
        let imitation = try recording(root: root)
        let trained = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: [imitation], preferences: preferences, resume: false),
            control: TrainingControl(), publish: trained.append, checkpointSaved: { _ in })
        XCTAssertEqual(trained.last?.phase, .complete, trained.last?.message ?? "Missing progress")
        XCTAssertTrue(trained.last?.loss?.isFinite ?? false)
        XCTAssertNil(trained.last?.validationLoss)
    }
}

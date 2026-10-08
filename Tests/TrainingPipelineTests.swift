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

    private func recording(root: URL, kind: LibraryKind = .imitation, includeActions: Bool = true,
                           availabilityLag: UInt64 = 0) throws -> RecordingItem {
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
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time + availabilityLag, sourceTimeNanoseconds: time,
                imageFile: file, width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)), state: InputState(), reusedPixels: false))
            if includeActions && index < 5 {
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

    func testLongInstructionsAreRejectedOnlyWhenConditioningIsEnabled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var item = try recording(root: root), c = configuration()
        item.edits.instruction = String(repeating: "é", count: 49)
        c.instructionConditioning = true
        XCTAssertThrowsError(try PreparedDataset.prepare(items: [item], configuration: c, settings: TrainingSettings(), stage: .imitation,
            root: root.appendingPathComponent("rejected"), checkCancellation: {}, progress: { _ in })) { error in
            XCTAssertTrue(error.localizedDescription.contains(item.name))
            XCTAssertTrue(error.localizedDescription.contains("98 UTF-8 bytes"))
        }
        XCTAssertThrowsError(try PolicyRunner(model: PolicyNetwork(configuration: c), permissions: c.capabilities, instruction: item.instruction))
        c.instructionConditioning = false
        let dataset = try PreparedDataset.prepare(items: [item], configuration: c, settings: TrainingSettings(), stage: .imitation,
            root: root.appendingPathComponent("accepted"), checkCancellation: {}, progress: { _ in })
        let batch = try TrainingBatch.load(plan: SequenceBatchPlan(recordings: dataset.training, chunk: 0), configuration: c,
            stage: .imitation, checkCancellation: {})
        XCTAssertEqual(sum(batch.arrays[BatchField.instructions.rawValue]).item(Int32.self), 0)
        XCTAssertEqual(item.instruction, String(repeating: "é", count: 49), "The original instruction is preserved for editing.")
    }

    func testPretrainingFingerprintIncludesTargetAtExactTrimEnd() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var item = try recording(root: root, kind: .pretraining)
        item.edits.trimEnd = 0.6
        func prepare(_ name: String) throws -> PreparedDataset {
            try PreparedDataset.prepare(items: [item], configuration: configuration(), settings: TrainingSettings(), stage: .pretraining,
                root: root.appendingPathComponent(name), checkCancellation: {}, progress: { _ in })
        }
        let before = try prepare("before")
        let examples = try before.training[0].examples(start: 0, count: before.exampleCount)
        XCTAssertEqual(examples.last?.nextObservation?.imageFile, "frames/5.jpg")
        XCTAssertFalse(examples.contains { $0.observation.imageFile == "frames/5.jpg" })
        let target = item.url.appendingPathComponent("frames/5.jpg")
        try Data(contentsOf: item.url.appendingPathComponent("frames/4.jpg")).write(to: target)
        let after = try prepare("after")
        XCTAssertEqual(before.exampleCount, after.exampleCount)
        XCTAssertNotEqual(before.fingerprint, after.fingerprint,
                          "Resume must detect a changed future target, even if no decision uses it as input.")
    }

    func testObservationOnlyPretrainingUsesChangingFramesDuringWaits() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root, kind: .pretraining, includeActions: false, availabilityLag: 20_000_000)
        let dataset = try PreparedDataset.prepare(items: [item], configuration: configuration(), settings: TrainingSettings(), stage: .pretraining,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        XCTAssertEqual(dataset.exampleCount, 5, "Every changing frame except the last has a future target.")
        XCTAssertEqual(dataset.nonWaitExampleCount, 0)
        let rows = try dataset.training[0].examples(start: 0, count: 5)
        XCTAssertTrue(rows.allSatisfy(\.hasCausalFuture))
        var row = try XCTUnwrap(rows.first)
        XCTAssertLessThan(try XCTUnwrap(row.nextObservation?.sourceTimeNanoseconds),
                          row.decisionTime + UInt64(row.targetDelay * 1e9), "Capture precedes wait completion/availability.")
        let batch = try TrainingBatch.load(plan: SequenceBatchPlan(recordings: dataset.training, chunk: 0),
            configuration: configuration(), stage: .pretraining, checkCancellation: {})
        XCTAssertEqual(sum(batch.arrays[BatchField.futureMask.rawValue]).item(Float.self), 2)

        row.targetAction = .keyDown(code: 0); row.targetDelay = 0.15
        XCTAssertFalse(row.hasCausalFuture, "Input actions still require pixels captured after the action.")
        row.targetAction = .wait(seconds: 0.1)
        row.nextObservation?.sourceTimeNanoseconds = row.decisionTime - 1
        XCTAssertFalse(row.hasCausalFuture, "A newly delivered but stale frame is not a future of this decision.")
        row.nextObservation?.reusedPixels = true
        XCTAssertTrue(row.hasCausalFuture, "Explicit static reuse remains supported.")
    }

    func testWorkerPauseResumeMatchesUninterruptedTrainingAndSavesValidation() throws {
        try checkWorkerPauseResume(architecture: .recurrent)
    }

    func testMidEpochCheckpointDoesNotInheritValidationOfEarlierWeights() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try [recording(root: root), recording(root: root)]
        var model = AIModel(name: "Checkpoint validation provenance"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 2; settings.batchSize = 1; settings.checkpointInterval = 100
        let preferences = AppPreferences.defaults(at: root), progress = ProgressCollector(), control = TrainingControl()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items,
            preferences: preferences, resume: false), control: control, publish: { update in
                progress.append(update)
                if update.phase == .training && update.epoch == 1 && update.cursor > 0 { control.set(.pause) }
            }, checkpointSaved: { _ in })
        XCTAssertEqual(progress.last?.phase, .paused, progress.last?.message ?? "Missing progress")
        // The UI can still show the last completed validation while training.
        XCTAssertNotNil(progress.last?.validationLoss)
        let store = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
        let latest = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration)).0
        XCTAssertGreaterThan(latest.sampleCursor, 0)
        XCTAssertNil(latest.validationLoss)
        XCTAssertNil(latest.actionEvaluation)
        XCTAssertNotNil(latest.bestValidationLoss)
        let best = try store.inference(modelID: model.id, latestID: latest.id,
            configuration: model.configuration, preferBest: true).0
        XCTAssertEqual(best.sampleCursor, 0)
        XCTAssertNotNil(best.validationLoss)
        XCTAssertEqual(best.actionEvaluation?.total, 11)
        XCTAssertLessThan(best.step, latest.step)
    }

    func testAttentionWorkerRestoresHistoryAcrossPauseResume() throws {
        try checkWorkerPauseResume(architecture: .attention)
    }

    func testInterruptedFinalValidationResumesWithoutRepeatingTraining() throws {
        for (request, expectedPhase) in [(TrainingControl.Request.pause, TrainingPhase.paused), (.cancel, .cancelled)] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let items = try [recording(root: root), recording(root: root)]
            var model = AIModel(name: "Interrupted validation"); model.configuration = configuration()
            var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1; settings.checkpointInterval = 100
            let preferences = AppPreferences.defaults(at: root), control = TrainingControl(), progress = ProgressCollector()
            TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items,
                preferences: preferences, resume: false), control: control, publish: { update in
                    progress.append(update)
                    if update.phase == .validating { control.set(request) }
                }, checkpointSaved: { _ in })
            XCTAssertEqual(progress.last?.phase, expectedPhase, progress.last?.message ?? "Missing progress")
            let store = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
            let interrupted = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
            XCTAssertEqual(interrupted.0.epoch, 0, "An unfinished validation pass must not advance the epoch.")
            XCTAssertGreaterThan(interrupted.0.sampleCursor, 0)
            XCTAssertNil(interrupted.0.validationLoss)
            XCTAssertNil(interrupted.0.actionEvaluation)
            let weightsBefore = try MLX.loadArrays(url: interrupted.1.appendingPathComponent("weights.safetensors"))
            let resumed = ProgressCollector()
            TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items,
                preferences: preferences, resume: true), control: TrainingControl(), publish: resumed.append, checkpointSaved: { _ in })
            XCTAssertEqual(resumed.last?.phase, .complete, resumed.last?.message ?? "Missing progress")
            let completed = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
            XCTAssertEqual(completed.0.epoch, 1)
            XCTAssertEqual(completed.0.sampleCursor, 0)
            XCTAssertEqual(completed.0.step, interrupted.0.step, "Resume should validate the same weights without another optimizer update.")
            XCTAssertNotNil(completed.0.validationLoss)
            XCTAssertEqual(completed.0.actionEvaluation?.total, 11, "Only a complete held-out pass can publish evaluation metrics.")
            let weightsAfter = try MLX.loadArrays(url: completed.1.appendingPathComponent("weights.safetensors"))
            XCTAssertEqual(Set(weightsAfter.keys), Set(weightsBefore.keys))
            for key in weightsBefore.keys {
                XCTAssertEqual(max(abs(weightsBefore[key]! - weightsAfter[key]!)).item(Float.self), 0, key)
            }
        }
    }

    func testStateBalancedWorkerResumeMatchesUninterruptedWeights() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceActionFrequency: true)
    }

    func testBalancedChoiceWorkerResumeMatchesUninterruptedWeights() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceInputChoices: true)
    }

    func testInterleavedRecurrentResumeRestoresEveryRecordingGroup() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceActionFrequency: true, scheduleVersion: 2, balanceVersion: 3, recordingCount: 7, pauseStep: 9)
    }

    func testInterleavedAttentionResumeRestoresEveryRecordingGroup() throws {
        try checkWorkerPauseResume(architecture: .attention, balanceActionFrequency: true, scheduleVersion: 2, balanceVersion: 3, recordingCount: 7, pauseStep: 9)
    }

    func testRotatedRecurrentResumeMatchesUninterruptedWeights() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceActionFrequency: true, scheduleVersion: 3, balanceVersion: 3, recordingCount: 7, pauseStep: 9)
    }

    func testRotatedAttentionResumeMatchesUninterruptedWeights() throws {
        try checkWorkerPauseResume(architecture: .attention, balanceActionFrequency: true, scheduleVersion: 3, balanceVersion: 3, recordingCount: 7, pauseStep: 9)
    }

    func testPauseDuringRotatedHistoryWarmupResumesExactly() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceActionFrequency: true, scheduleVersion: 3, balanceVersion: 3,
                                   recordingCount: 7, pauseDuringWarmup: true)
    }

    func testOlderRotatedCheckpointKeepsFixedGradientClippingOnResume() throws {
        try checkWorkerPauseResume(architecture: .recurrent, balanceActionFrequency: true, scheduleVersion: 3, balanceVersion: 3,
                                   recordingCount: 7, pauseStep: 9, weightedClip: nil, matchedControlSelection: nil)
    }

    func testPauseBeforeFirstWarmupUpdateDoesNotClaimANewCheckpoint() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root)
        var model = AIModel(name: "Initial warmup"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1
        let dataset = try PreparedDataset.prepare(items: [item], configuration: model.configuration, settings: settings,
            stage: .imitation, root: root.appendingPathComponent("probe"), checkCancellation: {}, progress: { _ in })
        settings.seed = try XCTUnwrap((0..<100).map(UInt64.init).first { seed in
            SequenceSchedule(recordings: dataset.training, batchSize: 1, sequenceLength: model.configuration.sequenceLength,
                             seed: seed, interleaved: true, rotated: true).plan(at: 0).map { $0.warmupChunks > 0 } ?? false
        })
        let preferences = AppPreferences.defaults(at: root), control = TrainingControl(), progress = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: [item], preferences: preferences, resume: false),
            control: control, publish: { update in
                progress.append(update)
                if update.message.hasPrefix("Warming recording history") { control.set(.pause) }
            }, checkpointSaved: { _ in XCTFail("No optimizer update occurred") })
        XCTAssertEqual(progress.last?.phase, .paused)
        XCTAssertEqual(progress.last?.step, 0)
        XCTAssertNil(progress.last?.checkpoint)
        XCTAssertTrue(progress.last?.message.contains("No new checkpoint") ?? false)
        XCTAssertNil(try CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
            .latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
    }

    private func checkWorkerPauseResume(architecture: TemporalArchitecture, balanceInputChoices: Bool = false, balanceActionFrequency: Bool = false,
                                       scheduleVersion: Int? = nil, balanceVersion: Int? = 2, recordingCount: Int = 2, pauseStep: Int = 1,
                                       pauseDuringWarmup: Bool = false, weightedClip: Bool? = true, matchedControlSelection: Bool? = true) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try (0..<recordingCount).map { _ in try recording(root: root) }
        var model = AIModel(name: "Training pipeline"); model.configuration = configuration()
        model.configuration.memory = architecture
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1; settings.checkpointInterval = 100
        settings.balancesInputChoices = balanceInputChoices
        settings.balancesActionFrequency = balanceActionFrequency
        settings.sequenceScheduleVersion = scheduleVersion; settings.actionBalanceVersion = balanceVersion
        settings.balancedGradientClipping = weightedClip
        settings.matchedControlCheckpoints = matchedControlSelection
        var preferences = AppPreferences.defaults(at: root)
        preferences.memoryLimitGB = 4; preferences.cacheLimitGB = 1
        let pause = TrainingControl(), first = ProgressCollector()
        let request = TrainingRequest(model: model, settings: settings, stage: .imitation, items: items, preferences: preferences, resume: false)
        TrainingWorker.run(request, control: pause, publish: { update in
            first.append(update)
            if update.phase == .training && (pauseDuringWarmup
                ? update.step > 0 && update.message.hasPrefix("Warming recording history")
                : update.step == pauseStep) { pause.set(.pause) }
        }, checkpointSaved: { _ in })
        XCTAssertEqual(first.last?.phase, .paused, first.last?.message ?? "Missing progress")
        if pauseDuringWarmup { XCTAssertTrue(first.last?.message.contains("memory warm-up") ?? false) }
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
        let resumedOptimizer = try MLX.loadArrays(url: resumedCheckpoint.1.appendingPathComponent("optimizer.safetensors"))
        let uninterrupted = ProgressCollector()
        TrainingWorker.run(request, control: TrainingControl(), publish: uninterrupted.append, checkpointSaved: { _ in })
        XCTAssertEqual(uninterrupted.last?.phase, .complete, uninterrupted.last?.message ?? "Missing progress")
        let fullCheckpoint = try XCTUnwrap(checkpoints.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        let fullWeights = try MLX.loadArrays(url: fullCheckpoint.1.appendingPathComponent("weights.safetensors"))
        let fullOptimizer = try MLX.loadArrays(url: fullCheckpoint.1.appendingPathComponent("optimizer.safetensors"))
        for key in resumedWeights.keys {
            XCTAssertLessThan(max(abs(resumedWeights[key]! - fullWeights[key]!)).item(Float.self), 1e-5, key)
        }
        XCTAssertEqual(resumedCheckpoint.0.step, fullCheckpoint.0.step)
        XCTAssertEqual(resumedCheckpoint.0.validationRecordingIDs, fullCheckpoint.0.validationRecordingIDs)
        XCTAssertEqual(resumedCheckpoint.0.bestHasMatchedControl, fullCheckpoint.0.bestHasMatchedControl)
        XCTAssertEqual(resumedCheckpoint.0.actionEvaluation?.initialPressTotal, 5)
        for key in resumedOptimizer.keys {
            XCTAssertLessThanOrEqual(max(abs(resumedOptimizer[key]! - fullOptimizer[key]!)).item(Float.self), 1e-5, key)
        }
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
        XCTAssertEqual(trained.last?.initialWeights, .checkpoint(id: checkpoint, stage: .pretraining))
    }

    func testFineTuningUsesTrainedWeightsWithFreshOptimizerAndChangedData() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try recording(root: root)
        var model = AIModel(name: "Fine-tuning pipeline"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1; settings.validationFraction = 0
        let preferences = AppPreferences.defaults(at: root), baseline = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: [original],
            preferences: preferences, resume: false), control: TrainingControl(), publish: baseline.append, checkpointSaved: { _ in })
        XCTAssertEqual(baseline.last?.phase, .complete, baseline.last?.message ?? "Missing progress")
        let store = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
        let parent = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        let originalPayload = try Data(contentsOf: parent.1.appendingPathComponent("weights.safetensors"))
        model.trainedCheckpoint = parent.0.id.uuidString; model.trainedFingerprint = model.configuration.fingerprint
        settings.startingWeights = .trained; settings.learningRate = 0.0001; settings.balancesActionFrequency = false
        let items = try [original, recording(root: root)], control = TrainingControl(), progress = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items,
            preferences: preferences, resume: false), control: control, publish: { update in
                progress.append(update)
                if update.phase == .training && update.step == 1 { control.set(.pause) }
            }, checkpointSaved: { _ in })
        XCTAssertEqual(progress.last?.phase, .paused, progress.last?.message ?? "Missing progress")
        let child = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        XCTAssertEqual(child.0.step, 1)
        XCTAssertEqual(child.0.epoch, 0)
        XCTAssertNotEqual(child.0.datasetFingerprint, parent.0.datasetFingerprint)
        XCTAssertEqual(Set(child.0.trainingRecordingIDs), Set(items.map(\.id)))
        XCTAssertEqual(child.0.initialWeights, .checkpoint(id: parent.0.id, stage: .imitation))
        XCTAssertEqual(try Data(contentsOf: parent.1.appendingPathComponent("weights.safetensors")), originalPayload)

        // Independently take the first update from the parent's weights. Matching
        // every tensor proves initialization, fresh moments and reset memory.
        let expected = PolicyNetwork(configuration: model.configuration)
        try TrainingWorker.restoreWeights(expected, from: parent.1)
        expected.train(true)
        let dataset = try PreparedDataset.prepare(items: items, configuration: model.configuration, settings: settings,
            stage: .imitation, root: root.appendingPathComponent("expected"), checkCancellation: {}, progress: { _ in })
        let schedule = SequenceSchedule(recordings: dataset.training, batchSize: settings.batchSize,
            sequenceLength: model.configuration.sequenceLength, seed: settings.seed,
            interleaved: settings.interleavesRecordings, rotated: settings.rotatesSequences)
        let plan = try XCTUnwrap(schedule.plan(at: 0))
        var hidden: [MLXArray] = []
        for chunk in 0..<plan.warmupChunks {
            let prefix = try TrainingBatch.load(plan: SequenceBatchPlan(recordings: plan.recordings, chunk: chunk),
                configuration: model.configuration, stage: .imitation, checkCancellation: {})
            hidden = PolicyLoss.forward(expected, prefix.arrays + hidden).hidden.map { stopGradient($0) }
            eval(hidden)
        }
        let batch = try TrainingBatch.load(plan: plan, configuration: model.configuration,
            stage: .imitation, checkCancellation: {})
        let gradient = valueAndGrad(model: expected) { model, arrays in PolicyLoss.values(model, arrays, stage: .imitation) }
        let (_, gradients) = gradient(expected, batch.arrays + hidden)
        let optimizer = ResumableAdamW(learningRate: settings.learningRate, weightDecay: settings.weightDecay)
        _ = optimizer.update(model: expected, gradients: gradients, clip: settings.gradientClip)
        let actual = try MLX.loadArrays(url: child.1.appendingPathComponent("weights.safetensors"))
        for (key, tensor) in expected.parameters().flattened() {
            XCTAssertLessThanOrEqual(max(abs(tensor - (try XCTUnwrap(actual[key])))).item(Float.self), 1e-6, key)
        }
        let savedOptimizer = try MLX.loadArrays(url: child.1.appendingPathComponent("optimizer.safetensors"))
        XCTAssertEqual(savedOptimizer["optimizer_step"]?.item(Int.self), 1)
        for (key, tensor) in optimizer.arrays() where key != "optimizer_step" {
            XCTAssertLessThanOrEqual(max(abs(tensor - (try XCTUnwrap(savedOptimizer[key])))).item(Float.self), 1e-6, key)
        }
    }
}

extension TrainingPipelineTests {
    func testMatchedControlCheckpointPreferencePreservesInitiationAndLegacyLossSelection() {
        XCTAssertTrue(TrainingWorker.prefersCheckpoint(loss: 10, matchedControl: true, bestLoss: 0.1,
            bestHasMatchedControl: false, requireMatchedControl: true))
        XCTAssertFalse(TrainingWorker.prefersCheckpoint(loss: 0.01, matchedControl: false, bestLoss: 10,
            bestHasMatchedControl: true, requireMatchedControl: true))
        XCTAssertTrue(TrainingWorker.prefersCheckpoint(loss: 5, matchedControl: true, bestLoss: 10,
            bestHasMatchedControl: true, requireMatchedControl: true))
        XCTAssertTrue(TrainingWorker.prefersCheckpoint(loss: 0.1, matchedControl: false, bestLoss: nil,
            bestHasMatchedControl: false, requireMatchedControl: true))
        XCTAssertFalse(TrainingWorker.prefersCheckpoint(loss: 10, matchedControl: true, bestLoss: 0.1,
            bestHasMatchedControl: false, requireMatchedControl: false))
        XCTAssertTrue(TrainingWorker.prefersCheckpoint(loss: 0.01, matchedControl: false, bestLoss: 10,
            bestHasMatchedControl: true, requireMatchedControl: false))
    }

    func testBalancedClippingUsesValidSampleImportanceAndPreservesLegacyLimits() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root), c = configuration(), settings = TrainingSettings()
        let dataset = try PreparedDataset.prepare(items: [item], configuration: c, settings: settings, stage: .imitation,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        let plan = SequenceBatchPlan(recordings: dataset.training, chunk: 5)
        let balanced = PolicyLoss.ActionBalance(frequencies: dataset.stateFrequencies, balanceTokens: true, recordings: dataset.training)
        let batch = try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation, actionBalance: balanced, checkCancellation: {})
        // Idle has five presses and one wait: the rare wait has weight 3.
        // There is one valid row and one padding row; two balanced loss families
        // carry expected unit weight, giving a clipping scale of 3 / 2.
        XCTAssertEqual(batch.validCount, 1)
        XCTAssertEqual(batch.gradientWeightScale, 1.5, accuracy: 1e-6)
        let legacy = PolicyLoss.ActionBalance(frequencies: dataset.stateFrequencies)
        XCTAssertEqual(try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation,
            actionBalance: legacy, checkCancellation: {}).gradientWeightScale, 1)
        XCTAssertEqual(try TrainingBatch.load(plan: plan, configuration: c, stage: .pretraining,
            actionBalance: balanced, checkCancellation: {}).gradientWeightScale, 1)
    }

    func testWeightedClippingRetainsImportanceInAdamMoments() throws {
        func moments(importance: Float) throws -> [String: MLXArray] {
            let model = Linear(1, 1, bias: false), optimizer = ResumableAdamW(learningRate: 0.001, weightDecay: 0)
            let gradients = ModuleParameters.unflattened([("weight", MLXArray([10 * importance], [1, 1]))])
            _ = optimizer.update(model: model, gradients: gradients, clip: importance)
            return optimizer.arrays()
        }
        let ordinary = try moments(importance: 1), rare = try moments(importance: 4)
        XCTAssertEqual(try XCTUnwrap(rare["first.weight"]).item(Float.self),
                       try XCTUnwrap(ordinary["first.weight"]).item(Float.self) * 4, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(rare["second.weight"]).item(Float.self),
                       try XCTUnwrap(ordinary["second.weight"]).item(Float.self) * 16, accuracy: 1e-6)
    }

    func testRotatedScheduleWarmsOnlyPastAndResetsAtWrap() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let indexes = try (0..<19).map { index in
            IndexedRecording(item: try recording(root: root), examplesURL: root, offsetsURL: root, count: (index % 4 + 2) * 3)
        }
        let schedule = SequenceSchedule(recordings: indexes, batchSize: 1, sequenceLength: 3, seed: 42, interleaved: true, rotated: true)
        var chunks: [Int: [Int]] = [:], pending: Set<Int> = []
        XCTAssertTrue(schedule.groupStarts.contains { $0 > 0 })
        for cursor in 0..<schedule.count {
            XCTAssertEqual(Set(schedule.pendingMemoryGroups(at: cursor)), pending)
            let plan = try XCTUnwrap(schedule.plan(at: cursor)), group = plan.memoryGroup
            let expected = (schedule.groupStarts[group] + plan.sequencePosition) % schedule.groupSteps[group]
            XCTAssertEqual(plan.chunk, expected)
            XCTAssertEqual(plan.warmupChunks, plan.sequencePosition == 0 ? plan.chunk : 0)
            XCTAssertEqual(plan.resetsMemory, plan.chunk == 0)
            chunks[group, default: []].append(plan.chunk)
            if plan.endsMemory { pending.remove(group) } else { pending.insert(group) }
            XCTAssertLessThanOrEqual(pending.count, SequenceSchedule.maximumActiveGroups)
        }
        for group in schedule.groups.indices { XCTAssertEqual(chunks[group]?.sorted(), Array(0..<schedule.groupSteps[group])) }
        XCTAssertTrue(pending.isEmpty)
    }

    func testPointerFilterChangesImitationFingerprintButNotPretraining() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root), c = configuration()
        var settings = TrainingSettings()
        func prepare(_ name: String, stage: TrainingStage) throws -> PreparedDataset {
            try PreparedDataset.prepare(items: [item], configuration: c, settings: settings, stage: stage,
                root: root.appendingPathComponent(name), checkCancellation: {}, progress: { _ in })
        }
        let unfiltered = try prepare("unfiltered", stage: .imitation), pretraining = try prepare("pretraining", stage: .pretraining)
        settings.ignoresPointerMovement = true
        XCTAssertNotEqual(try prepare("filtered", stage: .imitation).fingerprint, unfiltered.fingerprint)
        XCTAssertEqual(try prepare("same-pretraining", stage: .pretraining).fingerprint, pretraining.fingerprint)
        let pointerFiltered = try prepare("pointer-filtered", stage: .imitation)
        settings.ignoresKeyRepeats = true
        XCTAssertNotEqual(try prepare("both-filtered", stage: .imitation).fingerprint, pointerFiltered.fingerprint)
        XCTAssertEqual(try prepare("same-repeat-pretraining", stage: .pretraining).fingerprint, pretraining.fingerprint)
    }

    func testPointerFilterMasksMovementWithoutChangingVocabularyOrOtherControls() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root)
        var c = configuration(); c.capabilities.relativePointer = true
        var settings = TrainingSettings(); settings.ignoresPointerMovement = true; settings.ignoresKeyRepeats = true
        let allowed = settings.imitationCapabilities(c.capabilities)
        XCTAssertEqual(allowed.keys, c.capabilities.keys)
        XCTAssertEqual(allowed.buttons, c.capabilities.buttons)
        XCTAssertEqual(allowed.scrolling, c.capabilities.scrolling)
        let dataset = try PreparedDataset.prepare(items: [item], configuration: c, settings: settings, stage: .imitation,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        let batch = try TrainingBatch.load(plan: SequenceBatchPlan(recordings: dataset.training, chunk: 0),
            configuration: c, stage: .imitation, capabilities: allowed, checkCancellation: {})
        let codec = PolicyActionCodec(capabilities: c.capabilities), mask = batch.arrays[BatchField.actionMask.rawValue][0, 0]
        XCTAssertEqual(mask.size, codec.count)
        for action in [ComputerAction.pointer(x: 0, y: 0), .relativePointer(dx: 0, dy: 0)] {
            XCTAssertLessThan(mask[try XCTUnwrap(codec.token(for: action))].item(Float.self), -1e8)
        }
        let key = try XCTUnwrap(codec.token(for: .keyDown(code: 0)))
        XCTAssertEqual(mask[key].item(Float.self), 0)
        let repeatToken = try XCTUnwrap(codec.token(for: .keyRepeat(code: 0)))
        XCTAssertLessThan(batch.arrays[BatchField.actionMask.rawValue][0, 1, repeatToken].item(Float.self), -1e8)
        var held = InputState(); held.keys.insert(0)
        XCTAssertEqual(codec.mask(state: held, capabilities: c.capabilities)[repeatToken], 0)
        XCTAssertEqual(TrainingSettings().imitationCapabilities(c.capabilities), c.capabilities)
    }

    func testInterleavedResumeRejectsMissingCarryFromAnotherGroup() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try (0..<7).map { _ in try recording(root: root) }
        var model = AIModel(name: "Carry integrity"); model.configuration = configuration()
        var settings = TrainingSettings(); settings.epochs = 1; settings.batchSize = 1
        let preferences = AppPreferences.defaults(at: root), control = TrainingControl(), progress = ProgressCollector()
        let request = TrainingRequest(model: model, settings: settings, stage: .imitation, items: items, preferences: preferences, resume: false)
        TrainingWorker.run(request, control: control, publish: { p in
            progress.append(p); if p.phase == .training && p.step == 9 { control.set(.pause) }
        }, checkpointSaved: { _ in })
        XCTAssertEqual(progress.last?.phase, .paused)
        let store = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
        let (saved, directory) = try XCTUnwrap(store.latest(modelID: model.id, stage: .imitation, configuration: model.configuration))
        var state = try MLX.loadArrays(url: directory.appendingPathComponent("optimizer.safetensors"))
        let carry = try XCTUnwrap(state.keys.first { $0.hasPrefix("carry.") })
        state.removeValue(forKey: carry)
        var missing = saved; missing.id = UUID()
        _ = try store.save(missing, isBest: false) { url in
            try FileManager.default.copyItem(at: directory.appendingPathComponent("weights.safetensors"), to: url.appendingPathComponent("weights.safetensors"))
            try MLX.save(arrays: state, url: url.appendingPathComponent("optimizer.safetensors"))
        }
        let resumed = ProgressCollector()
        TrainingWorker.run(TrainingRequest(model: model, settings: settings, stage: .imitation, items: items, preferences: preferences, resume: true),
            control: TrainingControl(), publish: resumed.append, checkpointSaved: { _ in })
        XCTAssertEqual(resumed.last?.phase, .failed)
        XCTAssertTrue(resumed.last?.message.contains("temporal state") ?? false)
    }

    func testInterleavingPreservesOrderCoverageAndBoundedMemory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let indexes = try (0..<19).map { index in
            IndexedRecording(item: try recording(root: root), examplesURL: root, offsetsURL: root, count: (index % 4 + 1) * 3)
        }
        let schedule = SequenceSchedule(recordings: indexes, batchSize: 1, sequenceLength: 3, seed: 42, interleaved: true)
        var chunks: [UUID: [Int]] = [:], pending: Set<Int> = []
        for cursor in 0..<schedule.count {
            XCTAssertEqual(Set(schedule.pendingMemoryGroups(at: cursor)), pending)
            let plan = try XCTUnwrap(schedule.plan(at: cursor))
            for recording in plan.recordings { chunks[recording.item.id, default: []].append(plan.chunk) }
            if plan.endsMemory { pending.remove(plan.memoryGroup) } else { pending.insert(plan.memoryGroup) }
            XCTAssertLessThanOrEqual(pending.count, SequenceSchedule.maximumActiveGroups)
        }
        XCTAssertTrue(pending.isEmpty)
        for recording in indexes { XCTAssertEqual(chunks[recording.item.id], Array(0..<(recording.count / 3))) }
        XCTAssertNil(schedule.plan(at: -1)); XCTAssertNil(schedule.plan(at: schedule.count))
        XCTAssertTrue(schedule.pendingMemoryGroups(at: schedule.count).isEmpty)
        let firstGroups = (0..<8).compactMap { schedule.plan(at: $0)?.memoryGroup }
        XCTAssertEqual(firstGroups, Array(0..<8))
    }

    func testBalanceWeightsUseTrainingSplitAndReachEveryBatch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let items = try [recording(root: root), recording(root: root), recording(root: root)]
        let c = configuration()
        let dataset = try PreparedDataset.prepare(items: items, configuration: c, settings: TrainingSettings(), stage: .imitation,
            root: root.appendingPathComponent("index"), checkCancellation: {}, progress: { _ in })
        XCTAssertEqual(dataset.stateFrequencies.reduce(0) { $0 + $1.total }, dataset.exampleCount)
        XCTAssertEqual(dataset.stateFrequencies.reduce(0) { $0 + $1.inputs }, dataset.nonWaitExampleCount)
        let balance = PolicyLoss.ActionBalance(frequencies: dataset.stateFrequencies)
        let plan = SequenceBatchPlan(recordings: dataset.training, chunk: 0)
        let batch = try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation, actionBalance: balance, checkCancellation: {})
        let weights = batch.arrays[BatchField.actionWeights.rawValue].asArray(Float.self)
        XCTAssertEqual(weights[0], balance.forState(InputState()).inputWeight)
        XCTAssertEqual(weights[2], balance.forState(InputState(keys: [0])).inputWeight)
        XCTAssertGreaterThan(weights[0], 0)
    }
}

extension TrainingPipelineTests {
    func testKeyboardBatchNeutralizesCursorAndKeepsRecordedState() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try recording(root: root)
        var c = configuration(); c.detailCrop = true
        var settings = TrainingSettings(); settings.ignoresPointerMovement = true; settings.validationFraction = 0
        let data = try PreparedDataset.prepare(items: [item], configuration: c, settings: settings, stage: .imitation,
            root: root.appendingPathComponent("keyboard"), checkCancellation: {}, progress: { _ in })
        let plan = SequenceBatchPlan(recordings: data.training, chunk: 0)
        let keyboard = try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation, cursorIndependent: true, checkCancellation: {})
        let legacy = try TrainingBatch.load(plan: plan, configuration: c, stage: .imitation, checkCancellation: {})
        let context = keyboard.arrays[BatchField.context.rawValue]
        XCTAssertEqual(context[0, 0, 133].item(Float.self), 0.5)
        XCTAssertEqual(context[0, 0, 134].item(Float.self), 0.5)
        XCTAssertEqual(sum(abs(keyboard.arrays[BatchField.crops.rawValue])).item(Float.self), 0)
        XCTAssertGreaterThan(sum(abs(legacy.arrays[BatchField.crops.rawValue])).item(Float.self), 0)
        XCTAssertEqual(try data.training[0].examples(start: 0, count: 1)[0].state.cursorX, 0)
        var old = settings; old.cursorIndependentKeys = nil
        let legacyData = try PreparedDataset.prepare(items: [item], configuration: c, settings: old, stage: .imitation,
            root: root.appendingPathComponent("legacy"), checkCancellation: {}, progress: { _ in })
        XCTAssertNotEqual(data.fingerprint, legacyData.fingerprint)
        let raw = try PreparedDataset.prepare(items: [item], configuration: c, settings: settings, stage: .pretraining,
            root: root.appendingPathComponent("raw"), checkCancellation: {}, progress: { _ in })
        let oldRaw = try PreparedDataset.prepare(items: [item], configuration: c, settings: old, stage: .pretraining,
            root: root.appendingPathComponent("oldRaw"), checkCancellation: {}, progress: { _ in })
        XCTAssertEqual(raw.fingerprint, oldRaw.fingerprint)
    }
}

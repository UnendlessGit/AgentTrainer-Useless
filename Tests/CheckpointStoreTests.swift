import XCTest
@testable import AgentTrainer

final class CheckpointStoreTests: XCTestCase {
    func testInferenceDoesNotSelectStaleBestFromAnotherDatasetOrArchitecture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CheckpointStore(root: root), modelID = UUID(), config = PolicyConfiguration()
        func save(_ configuration: PolicyConfiguration, dataset: String, best: Bool, balance: Bool = false) throws -> UUID {
            var manifest = CheckpointManifest(modelID: modelID, configuration: configuration,
                configurationFingerprint: configuration.fingerprint, datasetFingerprint: dataset, trainingRecordingIDs: [],
                validationRecordingIDs: [], stage: .imitation, settings: TrainingSettings(), step: 1, epoch: 0,
                sampleCursor: 1, trainingLoss: 1)
            manifest.settings.balancesInputChoices = balance
            return try store.save(manifest, isBest: best) { url in
                try Data([1]).write(to: url.appendingPathComponent("weights.safetensors"))
                try Data([2]).write(to: url.appendingPathComponent("optimizer.safetensors"))
            }.id
        }
        let best = try save(config, dataset: "original", best: true)
        let changedData = try save(config, dataset: "different", best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: changedData, configuration: config, preferBest: true).0.id, changedData)
        let sameData = try save(config, dataset: "original", best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: sameData, configuration: config, preferBest: true).0.id, best)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: sameData, configuration: config, preferBest: false).0.id, sameData)
        var changed = config; changed.memoryDepth += 1
        let changedModel = try save(changed, dataset: "original", best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: changedModel, configuration: changed, preferBest: true).0.id, changedModel)
        let changedLoss = try save(config, dataset: "original", best: false, balance: true)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: changedLoss, configuration: config, preferBest: true).0.id, changedLoss)
    }

    func testCleanupProtectsReferencesRecentCopiesAndFailsClosedOnBadPointers() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CheckpointStore(root: root)
        var model = AIModel(name: "Cleanup fixture")
        var ids: [UUID] = []
        for index in 0..<7 {
            var metadata = CheckpointManifest(modelID: model.id, configuration: model.configuration,
                configurationFingerprint: model.configuration.fingerprint, datasetFingerprint: "test", trainingRecordingIDs: [],
                validationRecordingIDs: [], stage: .imitation, settings: TrainingSettings(), step: index + 1, epoch: index,
                sampleCursor: 0, trainingLoss: 1)
            metadata.createdAt = Date(timeIntervalSince1970: Double(index))
            _ = try store.save(metadata, isBest: index == 0) { url in
                try Data([1]).write(to: url.appendingPathComponent("weights.safetensors"))
                try Data([2]).write(to: url.appendingPathComponent("optimizer.safetensors"))
            }
            ids.append(metadata.id)
        }
        model.trainedCheckpoint = ids[1].uuidString
        let plan = try CheckpointMaintenance.review(root: root, models: [model])
        XCTAssertEqual(Set(plan.removable.map(\.lastPathComponent)), Set([ids[2], ids[3]].map(\.uuidString)))
        XCTAssertEqual(plan.retained, 5)
        XCTAssertTrue(plan.issues.isEmpty)
        try Data("broken".utf8).write(to: root.appendingPathComponent(model.id.uuidString).appendingPathComponent("best-imitation.json"))
        let protected = try CheckpointMaintenance.review(root: root, models: [model])
        XCTAssertTrue(protected.removable.isEmpty)
        XCTAssertEqual(protected.issues.count, 1)
    }

    func testFailureDoesNotDestroyPreviousCheckpointAndCorruptionIsRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CheckpointStore(root: root)
        let config = PolicyConfiguration()
        let metadata = CheckpointManifest(modelID: UUID(), configuration: config, configurationFingerprint: config.fingerprint,
            datasetFingerprint: "test", trainingRecordingIDs: [], validationRecordingIDs: [], stage: .imitation, settings: TrainingSettings(),
            step: 10, epoch: 1, sampleCursor: 10, trainingLoss: 0.5)
        let saved = try store.save(metadata, isBest: true) { url in
            try Data("weights".utf8).write(to: url.appendingPathComponent("weights.safetensors"))
            try Data("optimizer".utf8).write(to: url.appendingPathComponent("optimizer.safetensors"))
        }
        var failed = metadata; failed.id = UUID(); failed.step = 20
        XCTAssertThrowsError(try store.save(failed, isBest: true) { url in
            try Data("incomplete".utf8).write(to: url.appendingPathComponent("weights.safetensors"))
            throw DataIntegrityError.io("Injected disk write failure")
        })
        let loaded = try XCTUnwrap(store.latest(modelID: metadata.modelID, stage: .imitation, configuration: config))
        XCTAssertEqual(loaded.0.id, saved.id)
        XCTAssertEqual(loaded.0.step, 10)
        var invalidPosition = loaded.0; invalidPosition.epoch = -1
        try AtomicFile.encode(invalidPosition, to: loaded.1.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try store.load(modelID: metadata.modelID, checkpointID: metadata.id, configuration: config))
        XCTAssertThrowsError(try store.latestMetadata(modelID: metadata.modelID, stage: .imitation))
        try AtomicFile.encode(loaded.0, to: loaded.1.appendingPathComponent("manifest.json"))
        try Data("corrupt".utf8).write(to: loaded.1.appendingPathComponent("weights.safetensors"))
        XCTAssertThrowsError(try store.load(modelID: metadata.modelID, checkpointID: metadata.id, configuration: config))
    }

    func testArchitectureMismatchCannotLoad() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CheckpointStore(root: root)
        let config = PolicyConfiguration()
        let metadata = CheckpointManifest(modelID: UUID(), configuration: config, configurationFingerprint: config.fingerprint,
            datasetFingerprint: "test", trainingRecordingIDs: [], validationRecordingIDs: [], stage: .pretraining, settings: TrainingSettings(),
            step: 1, epoch: 0, sampleCursor: 1, trainingLoss: 1)
        _ = try store.save(metadata, isBest: false) { url in
            try Data([1]).write(to: url.appendingPathComponent("weights.safetensors"))
            try Data([2]).write(to: url.appendingPathComponent("optimizer.safetensors"))
        }
        var changed = config; changed.memorySize = 512
        XCTAssertThrowsError(try store.latest(modelID: metadata.modelID, stage: .pretraining, configuration: changed))
    }
}

extension CheckpointStoreTests {
    func testBestCheckpointCannotCrossTrainingRunsOrActionObjectives() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CheckpointStore(root: root), modelID = UUID(), c = PolicyConfiguration(), run = UUID()
        func save(runID: UUID, balanced: Bool, best: Bool) throws -> UUID {
            var metadata = CheckpointManifest(modelID: modelID, configuration: c, configurationFingerprint: c.fingerprint,
                datasetFingerprint: "same data", trainingRecordingIDs: [], validationRecordingIDs: [], stage: .imitation,
                settings: TrainingSettings(), step: 1, epoch: 1, sampleCursor: 0, trainingLoss: 1)
            metadata.trainingRunID = runID; metadata.settings.balancesActionFrequency = balanced
            return try store.save(metadata, isBest: best) { url in
                try Data([1]).write(to: url.appendingPathComponent("weights.safetensors"))
                try Data([2]).write(to: url.appendingPathComponent("optimizer.safetensors"))
            }.id
        }
        let best = try save(runID: run, balanced: true, best: true)
        let sameRun = try save(runID: run, balanced: true, best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: sameRun, configuration: c, preferBest: true).0.id, best)
        let newRun = try save(runID: UUID(), balanced: true, best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: newRun, configuration: c, preferBest: true).0.id, newRun)
        let differentLoss = try save(runID: run, balanced: false, best: false)
        XCTAssertEqual(try store.inference(modelID: modelID, latestID: differentLoss, configuration: c, preferBest: true).0.id, differentLoss)
    }
}

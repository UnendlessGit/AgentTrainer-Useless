import XCTest
@testable import AgentTrainer

final class CheckpointStoreTests: XCTestCase {
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

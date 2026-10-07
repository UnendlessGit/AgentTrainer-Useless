import XCTest
@testable import AgentTrainer

@MainActor final class WorkspaceStoreTests: XCTestCase {
    func testUnsupportedModelVersionIsReportedAndPreservedOnLoad() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(supportURL: root)
        await store.load()
        var model = try store.createModel(name: "Future model")
        model.schemaVersion = 99
        let url = store.modelRoot.appendingPathComponent(model.id.uuidString + ".json")
        try AtomicFile.encode(model, to: url)
        let before = try Data(contentsOf: url)
        await store.load()
        XCTAssertTrue(store.models.isEmpty)
        XCTAssertTrue(store.error?.contains("unsupported version 99") ?? false)
        XCTAssertEqual(try Data(contentsOf: url), before)
        XCTAssertThrowsError(try store.saveModel(model))
    }

    func testStaleConfigurationDraftPreservesNewerCheckpointReferences() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(supportURL: root)
        await store.load()
        var draft = try store.createModel(name: "Before training")
        var trained = draft
        trained.trainedCheckpoint = UUID().uuidString
        trained.pretrainedCheckpoint = UUID().uuidString
        trained.checkpointFingerprint = trained.configuration.fingerprint
        trained.trainedFingerprint = trained.configuration.fingerprint
        trained.pretrainedFingerprint = trained.configuration.fingerprint
        try store.saveModel(trained)
        draft.name = "Renamed after training"
        draft.imitationFolderIDs = [UUID()]
        draft.imitationRecordingIDs = [UUID()]
        draft.pretrainingFolderIDs = [UUID()]
        try store.saveModelConfiguration(draft)
        let saved = try AtomicFile.decode(AIModel.self, from: store.modelRoot.appendingPathComponent(draft.id.uuidString + ".json"))
        XCTAssertEqual(saved.name, draft.name)
        XCTAssertEqual(saved.imitationFolderIDs, draft.imitationFolderIDs)
        XCTAssertEqual(saved.imitationRecordingIDs, draft.imitationRecordingIDs)
        XCTAssertEqual(saved.pretrainingFolderIDs, draft.pretrainingFolderIDs)
        XCTAssertEqual(saved.createdAt.timeIntervalSince1970, trained.createdAt.timeIntervalSince1970, accuracy: 1)
        XCTAssertEqual(saved.trainedCheckpoint, trained.trainedCheckpoint)
        XCTAssertEqual(saved.pretrainedCheckpoint, trained.pretrainedCheckpoint)
        XCTAssertEqual(saved.trainedFingerprint, trained.trainedFingerprint)
        XCTAssertEqual(saved.pretrainedFingerprint, trained.pretrainedFingerprint)
        XCTAssertTrue(saved.canRun)
        draft.configuration.memorySize *= 2
        try store.saveModelConfiguration(draft)
        let changed = try XCTUnwrap(store.models.first)
        XCTAssertFalse(changed.canRun)
        XCTAssertFalse(changed.pretrainingCompatible)
        XCTAssertEqual(changed.trainedCheckpoint, trained.trainedCheckpoint)
        store.activeOperations.insert("training")
        XCTAssertThrowsError(try store.saveModelConfiguration(draft))
        store.activeOperations.remove("training")
        draft.id = UUID()
        XCTAssertThrowsError(try store.saveModelConfiguration(draft))
    }
}

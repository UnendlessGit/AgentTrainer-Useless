import XCTest
@testable import AgentTrainer

@MainActor final class WorkspaceStoreTests: XCTestCase {
    func testWindowReloadCannotRecoverJournalsDuringStorageMigration() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(supportURL: root)
        await store.load()
        let journal = try RecordingJournal(root: store.recordingRoot, manifest: RecordingManifest(
            name: "Unfinished", folderID: UUID(), kind: .imitation, target: CaptureTarget(), settings: RecordingSettings()))
        let manifestURL = journal.url.appendingPathComponent("manifest.json")
        let original = try Data(contentsOf: manifestURL)
        store.migrating = true
        await store.load()
        XCTAssertEqual(try Data(contentsOf: manifestURL), original, "A window reload must not recover the source while it is being copied.")
        XCTAssertTrue(store.recordings.isEmpty)
        XCTAssertThrowsError(try store.createFolder(name: "During move", kind: .imitation))
        store.migrating = false
        await store.load()
        XCTAssertEqual(store.recordings.first?.manifest.status, .interrupted)
        withExtendedLifetime(journal) {}
    }

    func testRecordingMigrationRefreshesURLsBeforeReleasingTheWorkspace() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(supportURL: root.appendingPathComponent("Support"))
        await store.load()
        let journal = try RecordingJournal(root: store.recordingRoot, manifest: RecordingManifest(
            name: "Move me", folderID: try XCTUnwrap(store.folders.first?.id), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings()))
        try journal.finish(at: 1_000_000_000)
        await store.load()
        let oldURL = try XCTUnwrap(store.recordings.first?.url)
        let destination = root.appendingPathComponent("Moved recordings")
        try await store.relocateStorage(\.recordingsPath, to: destination)
        let moved = try XCTUnwrap(store.recordings.first)
        XCTAssertEqual(moved.url.standardizedFileURL, destination.appendingPathComponent(oldURL.lastPathComponent).standardizedFileURL)
        XCTAssertEqual(moved.id, journal.snapshot.id)
        XCTAssertFalse(store.migrating)
        XCTAssertFalse(store.loading)
        XCTAssertTrue(store.canAccessWorkspace)
        var edits = moved.edits; edits.name = "Edited at destination"
        try store.editRecording(moved, edits: edits)
        XCTAssertTrue(FileManager.default.fileExists(atPath: moved.url.appendingPathComponent("edits.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldURL.appendingPathComponent("edits.json").path))
    }

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

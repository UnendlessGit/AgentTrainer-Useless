import XCTest
@testable import AgentTrainer

final class AppPreferencesTests: XCTestCase {
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        return root
    }

    @MainActor func testOptionalRecoveryPreservesCustomRootsAndOriginalFile() throws {
        let root = try fixture(), support = root.appendingPathComponent("Support")
        var saved = AppPreferences.defaults(at: root.appendingPathComponent("Custom storage"))
        saved.memoryLimitGB = Int.max; saved.cacheLimitGB = Int.min; saved.appearance = "Invalid"
        saved.shortcuts = ShortcutBindings(); saved.shortcuts?.emergency.keyCode = UInt32.max
        let url = support.appendingPathComponent("preferences.json")
        try AtomicFile.encode(saved, to: url)
        let original = try Data(contentsOf: url)
        let store = WorkspaceStore(supportURL: support)
        XCTAssertTrue(store.canAccessWorkspace)
        XCTAssertNoThrow(try store.preferences.validate())
        XCTAssertEqual(store.preferences.recordingsPath, saved.recordingsPath)
        XCTAssertEqual(store.preferences.modelsPath, saved.modelsPath)
        XCTAssertEqual(store.preferences.checkpointsPath, saved.checkpointsPath)
        XCTAssertEqual(store.preferences.memoryLimitGB, 12)
        XCTAssertEqual(store.preferences.cacheLimitGB, 2)
        XCTAssertNotNil(store.notice)
        XCTAssertEqual(try Data(contentsOf: url), original)
    }

    @MainActor func testUnsupportedOrUnreadablePreferencesNeverCreateReplacementWorkspace() async throws {
        for invalidSchema in [false, true] {
            let root = try fixture(), url = root.appendingPathComponent("preferences.json")
            if invalidSchema {
                var saved = AppPreferences.defaults(at: root); saved.schemaVersion = 999
                try AtomicFile.encode(saved, to: url)
            } else { try Data("{broken".utf8).write(to: url) }
            let original = try Data(contentsOf: url)
            let store = WorkspaceStore(supportURL: root)
            await store.load()
            XCTAssertFalse(store.canAccessWorkspace)
            XCTAssertThrowsError(try store.savePreferences(.defaults(at: root)))
            XCTAssertThrowsError(try store.createFolder(name: "Unsafe replacement", kind: .imitation))
            XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("Recordings").path))
            XCTAssertEqual(try Data(contentsOf: url), original)
        }
    }

    func testRejectsRelativeOverlappingAndAliasedStorageRoots() throws {
        let root = try fixture()
        var saved = AppPreferences.defaults(at: root)
        saved.modelsPath = "relative"
        XCTAssertThrowsError(try saved.validateStorage())
        saved.modelsPath = saved.recordingsPath + "/nested"
        XCTAssertThrowsError(try saved.validateStorage())
        try FileManager.default.createDirectory(atPath: saved.recordingsPath, withIntermediateDirectories: true)
        let alias = root.appendingPathComponent("Alias")
        try FileManager.default.createSymbolicLink(atPath: alias.path, withDestinationPath: saved.recordingsPath)
        saved.modelsPath = alias.path
        XCTAssertThrowsError(try saved.validateStorage())
        saved = .defaults(at: root); saved.recordingsPath = "/"
        XCTAssertThrowsError(try saved.validateStorage())
    }

    @MainActor func testInvalidPreferenceSaveCannotReplaceValidFile() throws {
        let root = try fixture(), store = WorkspaceStore(supportURL: root)
        try store.savePreferences(store.preferences)
        let url = root.appendingPathComponent("preferences.json"), original = try Data(contentsOf: url)
        var invalid = store.preferences; invalid.memoryLimitGB = Int.max
        XCTAssertThrowsError(try store.savePreferences(invalid))
        XCTAssertEqual(try Data(contentsOf: url), original)
        XCTAssertEqual(store.preferences.memoryLimitGB, 12)
    }

    @MainActor func testHumanInputSettingUpdatesTheSavedRunPolicyAndPreservesOtherControls() throws {
        let root = try fixture(), store = WorkspaceStore(supportURL: root)
        var preferences = store.preferences
        preferences.stopOnHumanInput = false
        try store.savePreferences(preferences)
        let runner = RunCoordinator(store: store)
        XCTAssertFalse(runner.configuration.stopOnHumanInput, "Legacy default applies before a Run setup exists.")
        runner.configuration.modelID = UUID()
        runner.configuration.maximumHoldSeconds = 4
        runner.configuration.instruction = "Keep this instruction"
        runner.saveConfiguration()
        let original = runner.configuration

        try runner.setHumanInputPolicy(true)
        var expected = original; expected.stopOnHumanInput = true
        XCTAssertEqual(runner.configuration, expected)
        XCTAssertEqual(RunCoordinator(store: store).configuration, expected,
                       "Restart must not restore an older Run override or the legacy default.")

        // Run's own checkbox is the same value read by Settings.
        runner.configuration.stopOnHumanInput = false
        runner.saveConfiguration()
        XCTAssertEqual(RunCoordinator(store: store).configuration, original)

        let file = root.appendingPathComponent("run-configuration.json")
        try FileManager.default.removeItem(at: file)
        try FileManager.default.createDirectory(at: file, withIntermediateDirectories: true)
        try Data([1]).write(to: file.appendingPathComponent("block-replacement"))
        XCTAssertThrowsError(try runner.setHumanInputPolicy(true))
        XCTAssertEqual(runner.configuration, original, "Failed saves must not appear applied in Settings.")
    }

    @MainActor func testVerifiedMigrationCanCommitWhileOtherWritesAreDisabled() async throws {
        let root = try fixture(), store = WorkspaceStore(supportURL: root.appendingPathComponent("Support"))
        await store.load()
        XCTAssertNil(store.error)
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.modelRoot.path), "Load must create the source directory: \(store.modelRoot.path)")
        let original = store.modelRoot.appendingPathComponent("fixture.bin")
        try Data([1, 3, 5, 7]).write(to: original)
        try AtomicFile.write(Data([2, 4, 6]), to: store.modelRoot.appendingPathComponent("Nested/data.bin"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: original.path))
        let destination = root.appendingPathComponent("Moved models")
        do { try await store.relocateStorage(\.modelsPath, to: destination) }
        catch { XCTFail("Relocation failed: \(error) source=\(original.path) destination=\(destination.path)"); return }
        XCTAssertFalse(store.migrating)
        XCTAssertEqual(store.modelRoot.standardizedFileURL, destination.standardizedFileURL)
        XCTAssertEqual(try Data(contentsOf: original), try Data(contentsOf: destination.appendingPathComponent("fixture.bin")))
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("Nested/data.bin")), Data([2, 4, 6]))
        let reopened = WorkspaceStore(supportURL: store.supportURL)
        XCTAssertEqual(reopened.preferences.modelsPath, destination.path)
    }

    @MainActor func testUnitHostUsesAnIsolatedWorkspaceWithoutGlobalShortcuts() throws {
        XCTAssertEqual(ProcessInfo.processInfo.environment["AGENTTRAINER_UNIT_TESTING"], "1")
        let session = AppSession()
        XCTAssertTrue(session.store.supportURL.lastPathComponent.hasPrefix("AgentTrainerUnitTests-"))
        session.installShortcuts()
        XCTAssertNil(session.store.error)
    }
}

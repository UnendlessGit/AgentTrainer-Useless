import Foundation
import Observation

@MainActor @Observable
final class WorkspaceStore {
    private(set) var preferences: AppPreferences
    private(set) var folders: [LibraryFolder] = []
    private(set) var recordings: [RecordingItem] = []
    private(set) var models: [AIModel] = []
    private(set) var loading = false
    var migrating = false
    var activeOperations: Set<String> = []
    var error: String?
    var notice: String?
    private var preferencesFailure: String?
    var canAccessWorkspace: Bool { preferencesFailure == nil }

    let supportURL: URL
    private var preferencesURL: URL { supportURL.appendingPathComponent("preferences.json") }
    private var foldersURL: URL { recordingRoot.appendingPathComponent("folders.json") }
    var recordingRoot: URL { URL(fileURLWithPath: preferences.recordingsPath, isDirectory: true) }
    var modelRoot: URL { URL(fileURLWithPath: preferences.modelsPath, isDirectory: true) }

    init(supportURL: URL? = nil) {
        self.supportURL = supportURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentTrainer", isDirectory: true)
        let preferenceFile = self.supportURL.appendingPathComponent("preferences.json")
        if FileManager.default.fileExists(atPath: preferenceFile.path) {
            do {
                var saved = try AtomicFile.decode(AppPreferences.self, from: preferenceFile)
                try saved.validateStorage()
                let repaired = saved.repairOptionalSettings()
                preferences = saved
                if !repaired.isEmpty {
                    notice = "Restored defaults for invalid \(repaired.joined(separator: ", ")). Your storage locations and original preferences file are preserved."
                }
            } catch {
                preferences = .defaults(at: self.supportURL)
                let message = "Preferences could not be read: \(error.localizedDescription) Restore \(preferenceFile.path) and reopen AgentTrainer. Workspace writes are disabled; the original file is preserved."
                preferencesFailure = message; self.error = message
            }
        } else { preferences = .defaults(at: self.supportURL) }
    }

    func load() async {
        guard canAccessWorkspace, !loading, activeOperations.isEmpty else { return }
        loading = true
        defer { loading = false }
        let recordingRoot = recordingRoot, modelRoot = modelRoot
        let checkpoints = CheckpointStore(root: URL(fileURLWithPath: preferences.checkpointsPath))
        do {
            let result = try await Task.detached(priority: .userInitiated) {
                try FileManager.default.createDirectory(at: recordingRoot, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
                let foldersURL = recordingRoot.appendingPathComponent("folders.json")
                let folders: [LibraryFolder]
                if FileManager.default.fileExists(atPath: foldersURL.path) {
                    folders = try AtomicFile.decode([LibraryFolder].self, from: foldersURL)
                } else {
                    folders = [LibraryFolder(name: "My demonstrations", kind: .imitation), LibraryFolder(name: "World observations", kind: .pretraining)]
                    try AtomicFile.encode(folders, to: foldersURL)
                }
                var recordings: [RecordingItem] = [], issues: [String] = []
                for url in try FileManager.default.contentsOfDirectory(at: recordingRoot, includingPropertiesForKeys: nil)
                    where url.pathExtension == "agentrecording" {
                    do {
                        let manifest = try RecordingJournal.recover(at: url)
                        let editsURL = url.appendingPathComponent("edits.json")
                        let edits = FileManager.default.fileExists(atPath: editsURL.path)
                            ? try AtomicFile.decode(RecordingEdits.self, from: editsURL) : RecordingEdits()
                        recordings.append(RecordingItem(manifest: manifest, edits: edits, url: url))
                    } catch { issues.append("\(url.lastPathComponent): \(error.localizedDescription)") }
                }
                var models: [AIModel] = []
                for url in try FileManager.default.contentsOfDirectory(at: modelRoot, includingPropertiesForKeys: nil)
                    where url.pathExtension == "json" {
                    do {
                        var model = try AtomicFile.decode(AIModel.self, from: url)
                        guard model.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(model.schemaVersion) }
                        // A crash after committing a checkpoint pointer but before
                        // saving the small model file must not strand completed work.
                        var recovered = false
                        for stage in TrainingStage.allCases {
                            do {
                                guard let saved = try checkpoints.latestMetadata(modelID: model.id, stage: stage),
                                      saved.configurationFingerprint == model.configuration.fingerprint,
                                      saved.configuration.fingerprint == model.configuration.fingerprint,
                                      saved.preprocessingVersion == 1, saved.actionCodecVersion == PolicyActionCodec.version else { continue }
                                if stage == .pretraining, model.pretrainedCheckpoint != saved.id.uuidString {
                                    model.pretrainedCheckpoint = saved.id.uuidString; model.pretrainedFingerprint = saved.configurationFingerprint; recovered = true
                                } else if stage == .imitation, model.trainedCheckpoint != saved.id.uuidString {
                                    model.trainedCheckpoint = saved.id.uuidString; model.trainedFingerprint = saved.configurationFingerprint; recovered = true
                                }
                            } catch { issues.append("\(model.name): \(error.localizedDescription)") }
                        }
                        if recovered { try AtomicFile.encode(model, to: url) }
                        models.append(model)
                    }
                    catch { issues.append("\(url.lastPathComponent): \(error.localizedDescription)") }
                }
                return (folders, recordings.sorted { $0.manifest.createdAt > $1.manifest.createdAt }, models.sorted { $0.createdAt < $1.createdAt }, issues)
            }.value
            folders = result.0; recordings = result.1; models = result.2
            if !result.3.isEmpty { error = "Some data needs attention:\n" + result.3.joined(separator: "\n") }
        } catch { self.error = error.localizedDescription }
    }

    func upsertRecording(_ manifest: RecordingManifest, url: URL) {
        if let index = recordings.firstIndex(where: { $0.id == manifest.id }) { recordings[index].manifest = manifest }
        else { recordings.insert(RecordingItem(manifest: manifest, edits: RecordingEdits(), url: url), at: 0) }
    }

    @discardableResult func createFolder(name: String, kind: LibraryKind, parentID: UUID? = nil) throws -> LibraryFolder {
        try requireWritable()
        let clean = try validatedName(name)
        if let parentID { guard folders.contains(where: { $0.id == parentID && $0.kind == kind }) else { throw DataIntegrityError.invalidData("Choose a parent in the same library section.") } }
        guard !folders.contains(where: { $0.name.localizedCaseInsensitiveCompare(clean) == .orderedSame && $0.kind == kind && $0.parentID == parentID }) else {
            throw DataIntegrityError.invalidData("A folder with this name already exists here.")
        }
        let folder = LibraryFolder(name: clean, kind: kind, parentID: parentID)
        let updated = folders + [folder]
        try AtomicFile.encode(updated, to: foldersURL)
        folders = updated
        return folder
    }

    func updateFolder(_ folder: LibraryFolder) throws {
        try requireWritable()
        var changed = folder
        changed.name = try validatedName(folder.name)
        guard !folders.contains(where: { $0.id != changed.id && $0.parentID == changed.parentID && $0.kind == changed.kind && $0.name.localizedCaseInsensitiveCompare(changed.name) == .orderedSame }) else {
            throw DataIntegrityError.invalidData("A folder with this name already exists here.")
        }
        var parent = changed.parentID, visited: Set<UUID> = [changed.id]
        while let id = parent {
            guard visited.insert(id).inserted, let found = folders.first(where: { $0.id == id && $0.kind == changed.kind }) else {
                throw DataIntegrityError.invalidData("A folder cannot be moved inside itself or to another library section.")
            }
            parent = found.parentID
        }
        let updated = folders.map { $0.id == changed.id ? changed : $0 }
        try AtomicFile.encode(updated, to: foldersURL)
        folders = updated
    }

    func deleteFolder(_ folder: LibraryFolder) throws {
        try requireWritable()
        guard !recordings.contains(where: { $0.manifest.folderID == folder.id }), !folders.contains(where: { $0.parentID == folder.id }) else {
            throw DataIntegrityError.invalidData("Move the recordings and subfolders before deleting this folder.")
        }
        let updated = folders.filter { $0.id != folder.id }
        try AtomicFile.encode(updated, to: foldersURL)
        folders = updated
    }

    func moveRecording(_ item: RecordingItem, to folderID: UUID) throws {
        try requireWritable()
        guard let folder = folders.first(where: { $0.id == folderID }), folder.kind == item.manifest.kind else {
            throw DataIntegrityError.invalidData("Choose a folder in the same library section.")
        }
        guard item.manifest.status != .recording else { throw DataIntegrityError.invalidData("Stop recording before moving it.") }
        var manifest = item.manifest
        manifest.folderID = folderID
        try AtomicFile.encode(manifest, to: item.url.appendingPathComponent("manifest.json"))
        upsertRecording(manifest, url: item.url)
    }

    func editRecording(_ item: RecordingItem, edits: RecordingEdits) throws {
        try requireWritable()
        if let name = edits.name { _ = try validatedName(name) }
        _ = try edits.timeRange(duration: item.manifest.duration)
        try AtomicFile.encode(edits, to: item.url.appendingPathComponent("edits.json"))
        if let index = recordings.firstIndex(where: { $0.id == item.id }) { recordings[index].edits = edits }
    }

    @discardableResult func createModel(name: String, copying original: AIModel? = nil) throws -> AIModel {
        var model = original ?? AIModel(name: name)
        model.id = UUID(); model.name = try validatedName(name); model.createdAt = Date(); model.modifiedAt = Date()
        // Duplication copies configuration/data selections, never claims ownership of another model's weights.
        model.checkpointFingerprint = nil; model.pretrainedCheckpoint = nil; model.trainedCheckpoint = nil
        model.pretrainedFingerprint = nil; model.trainedFingerprint = nil
        try saveModel(model)
        return model
    }

    func saveModel(_ model: AIModel) throws {
        try requireWritable()
        guard model.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(model.schemaVersion) }
        try model.configuration.validate()
        var updated = model
        updated.name = try validatedName(model.name); updated.modifiedAt = Date()
        try AtomicFile.encode(updated, to: modelRoot.appendingPathComponent(model.id.uuidString + ".json"))
        if let index = models.firstIndex(where: { $0.id == updated.id }) { models[index] = updated } else { models.append(updated) }
    }

    /// Editor drafts own configuration fields, not asynchronously saved checkpoints.
    /// Merge onto the current record so a stale editor cannot erase completed work.
    func saveModelConfiguration(_ draft: AIModel) throws {
        try requireWritable()
        guard activeOperations.isEmpty else {
            throw DataIntegrityError.invalidData("Finish the active operation before changing model configuration.")
        }
        guard var current = models.first(where: { $0.id == draft.id }) else {
            throw DataIntegrityError.invalidData("This model no longer exists. Select an existing model or create a new one.")
        }
        current.name = draft.name
        current.configuration = draft.configuration
        current.imitationFolderIDs = draft.imitationFolderIDs
        current.imitationRecordingIDs = draft.imitationRecordingIDs
        current.pretrainingFolderIDs = draft.pretrainingFolderIDs
        try saveModel(current)
    }

    func trashModel(_ model: AIModel) throws {
        try requireIdleMutation()
        let url = modelRoot.appendingPathComponent(model.id.uuidString + ".json")
        try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        models.removeAll { $0.id == model.id }
        notice = "Moved “\(model.name)” to Trash. Its checkpoints remain in storage."
    }

    func trashRecording(_ item: RecordingItem) throws {
        try requireIdleMutation()
        guard item.manifest.status != .recording else { throw DataIntegrityError.invalidData("Stop recording before moving it to Trash.") }
        try FileManager.default.trashItem(at: item.url, resultingItemURL: nil)
        recordings.removeAll { $0.id == item.id }
        notice = "Moved “\(item.name)” to Trash. Restore it to the recordings folder to use it again."
    }

    func reviewCheckpointStorage() async throws -> CheckpointCleanupPlan {
        let root = URL(fileURLWithPath: preferences.checkpointsPath), models = models
        return try await Task.detached(priority: .utility) { try CheckpointMaintenance.review(root: root, models: models) }.value
    }

    func cleanCheckpointStorage(oldCheckpoints: Bool) async throws {
        try requireIdleMutation()
        activeOperations.insert("maintenance")
        defer { activeOperations.remove("maintenance") }
        let root = URL(fileURLWithPath: preferences.checkpointsPath), models = models
        let count = try await Task.detached(priority: .utility) {
            if oldCheckpoints {
                // Re-review immediately before moving, rather than trusting a stale UI plan.
                let plan = try CheckpointMaintenance.review(root: root, models: models)
                for url in plan.removable { try FileManager.default.trashItem(at: url, resultingItemURL: nil) }
                return plan.removable.count
            }
            let cache = root.appendingPathComponent(".datasets")
            if FileManager.default.fileExists(atPath: cache.path) { try FileManager.default.removeItem(at: cache) }
            return 0
        }.value
        notice = oldCheckpoints ? "Moved \(count) older checkpoints to Trash. Empty Trash in Finder to reclaim their disk space."
            : "Temporary training indexes cleared. Recordings and checkpoints preserved."
    }

    private func requireIdleMutation() throws {
        try requireWritable()
        guard activeOperations.isEmpty else { throw DataIntegrityError.invalidData("Stop the active operation before removing workspace data.") }
    }

    func savePreferences(_ updated: AppPreferences) throws {
        // Relocation commits its verified destination while `migrating` is set.
        if let preferencesFailure { throw DataIntegrityError.io(preferencesFailure) }
        try updated.validate()
        try AtomicFile.encode(updated, to: preferencesURL)
        preferences = updated
    }

    func folderPath(_ folder: LibraryFolder) -> String {
        var names = [folder.name], parent = folder.parentID, seen: Set<UUID> = [folder.id]
        while let id = parent, seen.insert(id).inserted, let next = folders.first(where: { $0.id == id }) {
            names.insert(next.name, at: 0); parent = next.parentID
        }
        return names.joined(separator: " / ")
    }

    func perform(_ action: () throws -> Void) { do { try action() } catch { self.error = error.localizedDescription } }

    func requireWritable() throws {
        if let preferencesFailure { throw DataIntegrityError.io(preferencesFailure) }
        guard !migrating else { throw DataIntegrityError.io("Wait for the storage move to finish before changing the workspace.") }
    }

    private func validatedName(_ name: String) throws -> String {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty && clean.count <= 200 else { throw DataIntegrityError.invalidData("Use a name between 1 and 200 characters.") }
        return clean
    }
}

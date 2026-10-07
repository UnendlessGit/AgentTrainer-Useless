import Foundation
import Observation

@MainActor @Observable
final class WorkspaceStore {
    private(set) var preferences: AppPreferences
    private(set) var folders: [LibraryFolder] = []
    private(set) var recordings: [RecordingItem] = []
    private(set) var models: [AIModel] = []
    private(set) var loading = false
    var error: String?
    var notice: String?

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
            do { preferences = try AtomicFile.decode(AppPreferences.self, from: preferenceFile) }
            catch { preferences = .defaults; self.error = "Preferences could not be read: \(error.localizedDescription)" }
        } else { preferences = .defaults }
    }

    func load() async {
        guard !loading else { return }
        loading = true
        defer { loading = false }
        let recordingRoot = recordingRoot, modelRoot = modelRoot
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
                    do { models.append(try AtomicFile.decode(AIModel.self, from: url)) }
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
        guard !recordings.contains(where: { $0.manifest.folderID == folder.id }), !folders.contains(where: { $0.parentID == folder.id }) else {
            throw DataIntegrityError.invalidData("Move the recordings and subfolders before deleting this folder.")
        }
        let updated = folders.filter { $0.id != folder.id }
        try AtomicFile.encode(updated, to: foldersURL)
        folders = updated
    }

    func moveRecording(_ item: RecordingItem, to folderID: UUID) throws {
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
        if let name = edits.name { _ = try validatedName(name) }
        guard edits.trimStart.isFinite && edits.trimStart >= 0,
              (edits.trimEnd ?? item.manifest.duration).isFinite,
              edits.trimEnd ?? item.manifest.duration <= item.manifest.duration,
              (edits.trimEnd ?? item.manifest.duration) > edits.trimStart else {
            throw DataIntegrityError.invalidData("The trim must stay within the recording and have a positive duration.")
        }
        try AtomicFile.encode(edits, to: item.url.appendingPathComponent("edits.json"))
        if let index = recordings.firstIndex(where: { $0.id == item.id }) { recordings[index].edits = edits }
    }

    @discardableResult func createModel(name: String, copying original: AIModel? = nil) throws -> AIModel {
        var model = original ?? AIModel(name: name)
        model.id = UUID(); model.name = try validatedName(name); model.createdAt = Date(); model.modifiedAt = Date()
        // Duplication copies configuration/data selections, never claims ownership of another model's weights.
        model.checkpointFingerprint = nil; model.pretrainedCheckpoint = nil; model.trainedCheckpoint = nil
        try saveModel(model)
        return model
    }

    func saveModel(_ model: AIModel) throws {
        var updated = model
        updated.name = try validatedName(model.name); updated.modifiedAt = Date()
        try AtomicFile.encode(updated, to: modelRoot.appendingPathComponent(model.id.uuidString + ".json"))
        if let index = models.firstIndex(where: { $0.id == updated.id }) { models[index] = updated } else { models.append(updated) }
    }

    func savePreferences(_ updated: AppPreferences) throws {
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

    private func validatedName(_ name: String) throws -> String {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty && clean.count <= 200 else { throw DataIntegrityError.invalidData("Use a name between 1 and 200 characters.") }
        return clean
    }
}

import Foundation

struct CheckpointCleanupPlan: Sendable {
    var removable: [URL] = []
    var bytes: Int64 = 0
    var retained = 0
    var issues: [String] = []
}

/// Review immutable checkpoint metadata without loading tensors. Keep every
/// published pointer, every model reference, and three recent copies per stage.
enum CheckpointMaintenance {
    static func review(root: URL, models: [AIModel]) throws -> CheckpointCleanupPlan {
        var plan = CheckpointCleanupPlan()
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path) else { return plan }
        for directory in try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: .skipsHiddenFiles) {
            guard let modelID = UUID(uuidString: directory.lastPathComponent), try isDirectory(directory) else { continue }
            do {
                var protected = Set<UUID>()
                if let model = models.first(where: { $0.id == modelID }) {
                    for identifier in [model.pretrainedCheckpoint, model.trainedCheckpoint].compactMap({ $0 }).compactMap(UUID.init(uuidString:)) {
                        protected.insert(identifier)
                    }
                }
                for stage in TrainingStage.allCases {
                    for name in ["latest", "best"] {
                        let url = directory.appendingPathComponent("\(name)-\(stage.rawValue).json")
                        guard fm.fileExists(atPath: url.path) else { continue }
                        let pointer = try AtomicFile.decode(CheckpointPointer.self, from: url)
                        guard pointer.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(pointer.schemaVersion) }
                        protected.insert(pointer.checkpointID)
                    }
                }
                var checkpoints: [(CheckpointManifest, URL)] = []
                for child in try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: .skipsHiddenFiles) {
                    guard let id = UUID(uuidString: child.lastPathComponent), try isDirectory(child) else { continue }
                    let manifest = try AtomicFile.decode(CheckpointManifest.self, from: child.appendingPathComponent("manifest.json"))
                    guard manifest.schemaVersion == 1, manifest.id == id, manifest.modelID == modelID else {
                        throw DataIntegrityError.invalidData("Checkpoint metadata is inconsistent; this model was left untouched.")
                    }
                    checkpoints.append((manifest, child))
                }
                for stage in TrainingStage.allCases {
                    protected.formUnion(checkpoints.filter { $0.0.stage == stage }.sorted { $0.0.createdAt > $1.0.createdAt }.prefix(3).map { $0.0.id })
                }
                // Assemble per-model candidates only after all metadata validates.
                var candidates: [URL] = [], bytes: Int64 = 0
                for (manifest, url) in checkpoints where !protected.contains(manifest.id) {
                    bytes += try allocatedBytes(url); candidates.append(url)
                }
                plan.removable += candidates; plan.bytes += bytes
                plan.retained += checkpoints.count - candidates.count
            } catch { plan.issues.append("\(modelID.uuidString.prefix(8)): \(error.localizedDescription)") }
        }
        return plan
    }

    static func allocatedBytes(_ directory: URL) throws -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileAllocatedSizeKey]) else { return 0 }
        var size: Int64 = 0
        for case let file as URL in enumerator {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileAllocatedSizeKey])
            if values.isSymbolicLink == true { enumerator.skipDescendants(); continue }
            if values.isRegularFile == true { size += Int64(values.fileAllocatedSize ?? 0) }
        }
        return size
    }

    private static func isDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        return values.isDirectory == true && values.isSymbolicLink != true
    }
}

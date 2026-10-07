import Foundation
import CryptoKit

/// Synchronous operations called only on the model worker. Each checkpoint is an
/// immutable directory, committed before either latest/best pointer can name it.
struct CheckpointStore: Sendable {
    let root: URL

    func save(_ metadata: CheckpointManifest, isBest: Bool, writePayload: (URL) throws -> Void) throws -> CheckpointManifest {
        let modelRoot = root.appendingPathComponent(metadata.modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        let staging = modelRoot.appendingPathComponent(".\(metadata.id.uuidString).inprogress", isDirectory: true)
        let destination = modelRoot.appendingPathComponent(metadata.id.uuidString, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw DataIntegrityError.io("This checkpoint identifier is already in use.") }
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        try writePayload(staging)
        var manifest = metadata
        for name in ["weights.safetensors", "optimizer.safetensors"] {
            let url = staging.appendingPathComponent(name)
            let handle = try FileHandle(forWritingTo: url)
            do { try handle.synchronize(); try handle.close() } catch { try? handle.close(); throw error }
            manifest.files[name] = try digest(url)
        }
        guard manifest.configurationFingerprint == manifest.configuration.fingerprint else { throw DataIntegrityError.invalidData("Checkpoint configuration does not match its fingerprint.") }
        try AtomicFile.encode(manifest, to: staging.appendingPathComponent("manifest.json"))
        try AtomicFile.synchronizeDirectory(staging)
        try FileManager.default.moveItem(at: staging, to: destination)
        try AtomicFile.synchronizeDirectory(modelRoot)
        // A failure before this point cannot alter the previous valid pointer.
        try AtomicFile.encode(CheckpointPointer(checkpointID: manifest.id),
                              to: modelRoot.appendingPathComponent("latest-\(manifest.stage.rawValue).json"))
        if isBest {
            try AtomicFile.encode(CheckpointPointer(checkpointID: manifest.id),
                                  to: modelRoot.appendingPathComponent("best-\(manifest.stage.rawValue).json"))
        }
        return manifest
    }

    func load(modelID: UUID, checkpointID: UUID, configuration: PolicyConfiguration) throws -> (CheckpointManifest, URL) {
        let url = root.appendingPathComponent(modelID.uuidString).appendingPathComponent(checkpointID.uuidString)
        let manifest = try AtomicFile.decode(CheckpointManifest.self, from: url.appendingPathComponent("manifest.json"))
        guard manifest.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(manifest.schemaVersion) }
        guard manifest.modelID == modelID, manifest.id == checkpointID,
              manifest.configurationFingerprint == configuration.fingerprint,
              manifest.configuration.fingerprint == configuration.fingerprint,
              manifest.preprocessingVersion == 1, manifest.actionCodecVersion == PolicyActionCodec.version else {
            throw DataIntegrityError.invalidData("This checkpoint is incompatible with the current model configuration. Restore the original configuration or train a new model.")
        }
        guard manifest.epoch >= 0, manifest.sampleCursor >= 0, manifest.step >= 0 else {
            throw DataIntegrityError.invalidData("The checkpoint contains an invalid training position.")
        }
        for name in ["weights.safetensors", "optimizer.safetensors"] {
            guard let expected = manifest.files[name], try digest(url.appendingPathComponent(name)) == expected else {
                throw DataIntegrityError.invalidData("The checkpoint failed its integrity check (\(name)). Previous checkpoints remain preserved.")
            }
        }
        return (manifest, url)
    }

    func latest(modelID: UUID, stage: TrainingStage, configuration: PolicyConfiguration) throws -> (CheckpointManifest, URL)? {
        let pointerURL = root.appendingPathComponent(modelID.uuidString).appendingPathComponent("latest-\(stage.rawValue).json")
        guard FileManager.default.fileExists(atPath: pointerURL.path) else { return nil }
        let pointer = try AtomicFile.decode(CheckpointPointer.self, from: pointerURL)
        guard pointer.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(pointer.schemaVersion) }
        let loaded = try load(modelID: modelID, checkpointID: pointer.checkpointID, configuration: configuration)
        guard loaded.0.stage == stage else { throw DataIntegrityError.invalidData("The checkpoint stage does not match its pointer.") }
        return loaded
    }

    /// Small UI metadata read only. Weight integrity and configuration are checked
    /// again on the worker before any checkpoint is loaded for execution/training.
    func latestMetadata(modelID: UUID, stage: TrainingStage) throws -> CheckpointManifest? {
        let modelRoot = root.appendingPathComponent(modelID.uuidString)
        let pointerURL = modelRoot.appendingPathComponent("latest-\(stage.rawValue).json")
        guard FileManager.default.fileExists(atPath: pointerURL.path) else { return nil }
        let pointer = try AtomicFile.decode(CheckpointPointer.self, from: pointerURL)
        let manifest = try AtomicFile.decode(CheckpointManifest.self, from: modelRoot.appendingPathComponent(pointer.checkpointID.uuidString).appendingPathComponent("manifest.json"))
        guard pointer.schemaVersion == 1, manifest.schemaVersion == 1, manifest.id == pointer.checkpointID,
              manifest.modelID == modelID, manifest.stage == stage,
              manifest.epoch >= 0, manifest.sampleCursor >= 0, manifest.step >= 0 else { throw DataIntegrityError.invalidData("The checkpoint metadata is inconsistent.") }
        return manifest
    }

    func inference(modelID: UUID, latestID: UUID, configuration: PolicyConfiguration, preferBest: Bool) throws -> (CheckpointManifest, URL) {
        let modelRoot = root.appendingPathComponent(modelID.uuidString)
        var selected = latestID
        let bestURL = modelRoot.appendingPathComponent("best-imitation.json")
        if preferBest && FileManager.default.fileExists(atPath: bestURL.path) {
            let pointer = try AtomicFile.decode(CheckpointPointer.self, from: bestURL)
            guard pointer.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(pointer.schemaVersion) }
            let best = try AtomicFile.decode(CheckpointManifest.self, from: modelRoot.appendingPathComponent(pointer.checkpointID.uuidString).appendingPathComponent("manifest.json"))
            let latest = try AtomicFile.decode(CheckpointManifest.self, from: modelRoot.appendingPathComponent(latestID.uuidString).appendingPathComponent("manifest.json"))
            guard best.modelID == modelID, best.id == pointer.checkpointID, best.stage == .imitation else {
                throw DataIntegrityError.invalidData("The best checkpoint pointer is inconsistent.")
            }
            // A prior run's best pointer may remain after configuration/data changes
            // or a new run without validation. It must not override current weights.
            if best.configurationFingerprint == configuration.fingerprint && best.datasetFingerprint == latest.datasetFingerprint {
                selected = best.id
            }
        }
        let result = try load(modelID: modelID, checkpointID: selected, configuration: configuration)
        guard result.0.stage == .imitation else { throw DataIntegrityError.invalidData("Only imitation-learning checkpoints can control input.") }
        return result
    }

    private func digest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

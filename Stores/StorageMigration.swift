import Foundation
import CryptoKit

extension WorkspaceStore {
    func relocateStorage(_ keyPath: WritableKeyPath<AppPreferences, String>, to destination: URL) async throws {
        try requireWritable()
        guard activeOperations.isEmpty else { throw DataIntegrityError.invalidData("Stop recording, training and runs before changing storage locations.") }
        migrating = true
        defer { migrating = false }
        let source = URL(fileURLWithPath: preferences[keyPath: keyPath]).standardizedFileURL.resolvingSymlinksInPath()
        let destination = destination.standardizedFileURL.resolvingSymlinksInPath()
        guard source != destination else { return }
        guard !destination.path.hasPrefix(source.path + "/"), !source.path.hasPrefix(destination.path + "/") else {
            throw DataIntegrityError.invalidData("Choose a location outside the original storage folder.")
        }
        let otherRoots = [preferences.recordingsPath, preferences.modelsPath, preferences.checkpointsPath].filter { $0 != source.path }
        guard !otherRoots.contains(where: { destination.path == $0 || destination.path.hasPrefix($0 + "/") || $0.hasPrefix(destination.path + "/") }) else {
            throw DataIntegrityError.invalidData("Recordings, models and checkpoints need separate storage folders.")
        }
        var updated = preferences
        updated[keyPath: keyPath] = destination.path
        try updated.validate()
        try await Task.detached(priority: .utility) {
            let fm = FileManager.default
            try fm.createDirectory(at: destination, withIntermediateDirectories: true)
            let existing = try fm.contentsOfDirectory(at: destination, includingPropertiesForKeys: nil).filter { $0.lastPathComponent != ".DS_Store" }
            guard existing.isEmpty else { throw DataIntegrityError.invalidData("The destination folder is not empty. Choose an empty folder to prevent data from being overwritten.") }
            guard fm.fileExists(atPath: source.path) else { return }
            for file in try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) where file.lastPathComponent != ".DS_Store" {
                try fm.copyItem(at: file, to: destination.appendingPathComponent(file.lastPathComponent))
            }
            // Streaming hashes verify content without loading large recordings into memory.
            var enumerationError: Error?
            guard let enumerator = fm.enumerator(at: source, includingPropertiesForKeys: [.isRegularFileKey], errorHandler: { _, error in
                enumerationError = error; return false
            }) else { throw DataIntegrityError.io("The source storage folder could not be enumerated for verification.") }
            while let entry = enumerator.nextObject() as? URL {
                // Foundation may enumerate /var via /private/var even when its
                // root URL uses the shorter spelling. Normalize both before
                // deriving a relative path; never slice an unrelated prefix.
                let file = entry.standardizedFileURL
                guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true, file.lastPathComponent != ".DS_Store" else { continue }
                guard file.path.hasPrefix(source.path + "/") else { throw DataIntegrityError.io("A copied file could not be matched to its source folder.") }
                let relative = String(file.path.dropFirst(source.path.count + 1))
                guard try StorageDigest.hash(file) == StorageDigest.hash(destination.appendingPathComponent(relative)) else {
                    throw DataIntegrityError.io("The copied data did not verify. The original location remains active.")
                }
            }
            if let enumerationError { throw enumerationError }
            try AtomicFile.synchronizeDirectory(destination)
        }.value
        // Preserve appearance/resource changes made while the copy was running.
        updated = preferences
        updated[keyPath: keyPath] = destination.path
        try savePreferences(updated)
        await load()
        notice = "Storage updated. The original data remains at \(source.path)."
    }
}

private enum StorageDigest {
    static func hash(_ url: URL) throws -> SHA256.Digest {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty { hash.update(data: chunk) }
        return hash.finalize()
    }
}

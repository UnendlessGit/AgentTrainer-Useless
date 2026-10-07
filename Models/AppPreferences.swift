import Foundation

struct AppPreferences: Codable, Sendable {
    static let appearances = ["System", "Light", "Dark"]
    static let memoryLimitsGB = [4, 8, 12, 16, 24]
    static let cacheLimitsGB = [0, 1, 2, 4, 8]
    var schemaVersion = 1
    var recordingsPath: String
    var modelsPath: String
    var checkpointsPath: String
    var appearance = "System"
    var memoryLimitGB = 12
    var cacheLimitGB = 2
    var stopOnHumanInput = true
    var shortcuts: ShortcutBindings?

    func validateStorage() throws {
        guard schemaVersion == 1 else { throw DataIntegrityError.invalidData("This preferences version is not supported.") }
        let paths = [recordingsPath, modelsPath, checkpointsPath]
        guard paths.allSatisfy({ $0.hasPrefix("/") && !$0.contains("\0") }) else {
            throw DataIntegrityError.invalidData("Storage locations must be absolute folder paths.")
        }
        let roots = paths.map { URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath().path }
        for index in roots.indices {
            guard roots[index] != "/" else { throw DataIntegrityError.invalidData("The filesystem root cannot be a storage folder.") }
            for other in roots.indices where other != index {
                guard roots[index] != roots[other], !roots[index].hasPrefix(roots[other] + "/") else {
                    throw DataIntegrityError.invalidData("Recordings, models and checkpoints need separate, non-overlapping storage folders.")
                }
            }
        }
    }

    func validate() throws {
        try validateStorage()
        guard Self.appearances.contains(appearance), Self.memoryLimitsGB.contains(memoryLimitGB), Self.cacheLimitsGB.contains(cacheLimitGB) else {
            throw DataIntegrityError.invalidData("Choose supported appearance and memory limits in Settings.")
        }
        try shortcuts?.validate()
    }

    /// Repair only optional settings. Never redirect a saved storage location or
    /// overwrite the original preferences during launch-time recovery.
    mutating func repairOptionalSettings() -> [String] {
        var repaired: [String] = []
        if !Self.appearances.contains(appearance) { appearance = "System"; repaired.append("appearance") }
        if !Self.memoryLimitsGB.contains(memoryLimitGB) { memoryLimitGB = 12; repaired.append("MLX memory limit") }
        if !Self.cacheLimitsGB.contains(cacheLimitGB) { cacheLimitGB = 2; repaired.append("MLX cache limit") }
        do { try shortcuts?.validate() }
        catch { shortcuts = nil; repaired.append("keyboard shortcuts") }
        return repaired
    }

    static var defaults: AppPreferences {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentTrainer", isDirectory: true)
        return defaults(at: base)
    }

    static func defaults(at base: URL) -> AppPreferences {
        return AppPreferences(recordingsPath: base.appendingPathComponent("Recordings").path,
                              modelsPath: base.appendingPathComponent("Models").path,
                              checkpointsPath: base.appendingPathComponent("Checkpoints").path)
    }
}

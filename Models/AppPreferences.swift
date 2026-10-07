import Foundation

struct AppPreferences: Codable, Sendable {
    var schemaVersion = 1
    var recordingsPath: String
    var modelsPath: String
    var checkpointsPath: String
    var appearance = "System"
    var memoryLimitGB = 12
    var cacheLimitGB = 2
    var stopOnHumanInput = true
    var shortcuts: ShortcutBindings?

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

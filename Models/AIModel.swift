import Foundation
import CryptoKit

enum TemporalArchitecture: String, Codable, CaseIterable, Identifiable, Sendable {
    case recurrent = "Recurrent memory", attention = "Causal attention"
    var id: String { rawValue }
}

struct PolicyConfiguration: Codable, Equatable, Sendable {
    var schemaVersion = 1
    var imageSize = 224
    var patchSize = 16
    var visualWidth = 128
    var visualDepth = 3
    var memory: TemporalArchitecture = .recurrent
    var memorySize = 256
    var memoryDepth = 2
    var sequenceLength = 32
    var detailCrop = true
    var instructionConditioning = true
    var capabilities = ActionCapabilities()

    var fingerprint: String {
        // Sets have no stable Codable order. Encode capabilities as sorted arrays.
        let stable = [String(schemaVersion), String(ComputerAction.schemaVersion), String(imageSize), String(patchSize),
                      String(visualWidth), String(visualDepth), memory.rawValue, String(memorySize), String(memoryDepth),
                      String(sequenceLength), String(detailCrop), String(instructionConditioning),
                      capabilities.keys.sorted().map(String.init).joined(separator: ","),
                      capabilities.buttons.sorted().map(String.init).joined(separator: ","),
                      String(capabilities.pointer), String(capabilities.relativePointer), String(capabilities.scrolling),
                      String(capabilities.dragging), String(capabilities.chords), String(capabilities.maximumHeldKeys)]
        return SHA256.hash(data: Data(stable.joined(separator: "|").utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

struct AIModel: Codable, Identifiable, Sendable {
    var schemaVersion = 1
    var id = UUID()
    var name: String
    var createdAt = Date()
    var modifiedAt = Date()
    var configuration = PolicyConfiguration()
    var imitationFolderIDs: Set<UUID> = []
    var imitationRecordingIDs: Set<UUID> = []
    var pretrainingFolderIDs: Set<UUID> = []
    var checkpointFingerprint: String?
    var pretrainedCheckpoint: String?
    var trainedCheckpoint: String?
    var compatibility: String {
        guard let checkpointFingerprint else { return "Not trained" }
        return checkpointFingerprint == configuration.fingerprint ? "Compatible" : "Configuration changed · retraining required"
    }
    var canRun: Bool { trainedCheckpoint != nil && checkpointFingerprint == configuration.fingerprint }
}

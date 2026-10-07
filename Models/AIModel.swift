import Foundation
import CryptoKit

enum TemporalArchitecture: String, Codable, CaseIterable, Identifiable, Sendable {
    case recurrent = "Recurrent memory", attention = "Causal attention"
    var id: String { rawValue }
}

struct PolicyConfiguration: Codable, Equatable, Sendable {
    static let implementationVersion = 2
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

    func validate() throws {
        guard schemaVersion == 1, (64...512).contains(imageSize), (8...32).contains(patchSize), imageSize % patchSize == 0,
              (32...512).contains(visualWidth), visualWidth % 4 == 0, (1...8).contains(visualDepth),
              (32...1024).contains(memorySize), memorySize % 4 == 0, (1...4).contains(memoryDepth),
              (2...128).contains(sequenceLength), capabilities.keys.allSatisfy({ $0 < 128 }),
              capabilities.buttons.allSatisfy({ (0..<5).contains($0) }), (1...16).contains(capabilities.maximumHeldKeys) else {
            throw DataIntegrityError.invalidData("The model architecture or action capabilities are outside supported limits.")
        }
    }

    func estimatedWorkingSetBytes(batchSize: Int) -> Int {
        let patches = imageSize / patchSize * (imageSize / patchSize)
        let visualActivations = batchSize * sequenceLength * visualDepth * (detailCrop ? 2 : 1)
            * (4 * patches * patches * 4 + patches * visualWidth * 64)
        let parameters = visualDepth * visualWidth * visualWidth * 12 + memoryDepth * memorySize * memorySize * 12
        return visualActivations + parameters * 24 + 128 * 1_048_576
    }

    var fingerprint: String {
        // Sets have no stable Codable order. Encode capabilities as sorted arrays.
        let stable = [String(Self.implementationVersion), String(schemaVersion), String(ComputerAction.schemaVersion), String(imageSize), String(patchSize),
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
    var pretrainedFingerprint: String?
    var trainedFingerprint: String?
    var pretrainedCheckpoint: String?
    var trainedCheckpoint: String?
    var compatibility: String {
        guard trainedCheckpoint != nil || pretrainedCheckpoint != nil else { return "Not trained" }
        if trainedCheckpoint != nil { return canRun ? "Compatible" : "Configuration changed · retraining required" }
        return pretrainingCompatible ? "Pre-trained · ready for imitation learning" : "Pre-training configuration changed · new weights required"
    }
    var canRun: Bool { trainedCheckpoint != nil && (trainedFingerprint ?? checkpointFingerprint) == configuration.fingerprint }
    var pretrainingCompatible: Bool { pretrainedCheckpoint != nil && (pretrainedFingerprint ?? checkpointFingerprint) == configuration.fingerprint }
}

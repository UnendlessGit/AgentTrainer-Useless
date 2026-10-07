import Foundation

enum TrainingStage: String, Codable, CaseIterable, Sendable { case pretraining, imitation }

struct TrainingSettings: Codable, Equatable, Sendable {
    var epochs = 10
    var batchSize = 2
    var learningRate: Float = 0.0003
    var weightDecay: Float = 0.01
    var gradientClip: Float = 1
    var checkpointInterval = 100
    var validationFraction = 0.2
    var seed: UInt64 = 42
    /// Each run is bounded, per the acceptance objective. Resumable checkpoints
    /// let the user continue with additional explicitly started runs.
    var maximumRunMinutes = 20

    func validate() throws {
        guard (1...1000).contains(epochs), (1...16).contains(batchSize), learningRate.isFinite,
              learningRate > 0 && learningRate <= 0.1, weightDecay.isFinite && (0...1).contains(weightDecay),
              gradientClip.isFinite && gradientClip > 0, checkpointInterval > 0,
              validationFraction.isFinite && (0...0.5).contains(validationFraction), (1...30).contains(maximumRunMinutes) else {
            throw DataIntegrityError.invalidData("Use valid training settings and a time budget from 1 to 30 minutes.")
        }
    }
}

struct CheckpointManifest: Codable, Sendable {
    var schemaVersion = 1
    var id = UUID()
    var modelID: UUID
    var configuration: PolicyConfiguration
    var configurationFingerprint: String
    var preprocessingVersion = 1
    var actionCodecVersion = PolicyActionCodec.version
    var datasetFingerprint: String
    var trainingRecordingIDs: [UUID]
    var validationRecordingIDs: [UUID]
    var stage: TrainingStage
    var settings: TrainingSettings
    var step: Int
    var epoch: Int
    var sampleCursor: Int
    var trainingLoss: Float
    var validationLoss: Float?
    var bestValidationLoss: Float?
    var actionEvaluation: ActionEvaluation?
    var createdAt = Date()
    var files: [String: String] = [:]
}

struct CheckpointPointer: Codable, Sendable {
    var schemaVersion = 1
    var checkpointID: UUID
}

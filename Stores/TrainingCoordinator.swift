import Foundation
import Observation

@MainActor @Observable
final class TrainingCoordinator {
    private(set) var progress = TrainingProgress()
    var settings = TrainingSettings()
    var selectedModelID: UUID?
    private var control: TrainingControl?
    private let store: WorkspaceStore
    private var generation = UUID()
    private(set) var lastRequest: TrainingRequest?
    var isBusy: Bool { progress.phase.isBusy }

    init(store: WorkspaceStore) { self.store = store }

    func recordings(for model: AIModel, stage: TrainingStage) -> [RecordingItem] {
        store.recordings.filter { item in
            guard item.eligible else { return false }
            if stage == .pretraining { return item.manifest.kind == .pretraining && model.pretrainingFolderIDs.contains(item.manifest.folderID) }
            return item.manifest.kind == .imitation && (model.imitationFolderIDs.contains(item.manifest.folderID) || model.imitationRecordingIDs.contains(item.id))
        }
    }

    func start(model: AIModel, stage: TrainingStage, resume: Bool = false) {
        guard !isBusy else { return }
        guard !store.migrating, store.activeOperations.isEmpty else { store.error = "Finish the active operation before training."; return }
        store.activeOperations.insert("training")
        let request = TrainingRequest(model: model, settings: settings, stage: stage, items: recordings(for: model, stage: stage),
                                      preferences: store.preferences, resume: resume)
        let control = TrainingControl(), generation = UUID()
        self.control = control; self.generation = generation; lastRequest = request
        progress = TrainingProgress(phase: .preparing, message: "Preparing selected recordings…", modelID: model.id, stage: stage)
        TrainingWorker.queue.async { [self] in
            TrainingWorker.run(request, control: control) { update in
                DispatchQueue.main.async { [self] in
                    guard self.generation == generation else { return }
                    self.progress = update
                    if !update.phase.isBusy { self.store.activeOperations.remove("training") }
                    if update.phase == .failed { self.store.error = update.message }
                }
            } checkpointSaved: { manifest in
                DispatchQueue.main.async { [self] in
                    guard self.generation == generation,
                          var model = self.store.models.first(where: { $0.id == manifest.modelID }),
                          model.configuration.fingerprint == manifest.configurationFingerprint else { return }
                    // Preserve legacy fingerprints separately before either stage
                    // changes, so updating one cannot validate the other's weights.
                    model.trainedFingerprint = model.trainedFingerprint ?? model.checkpointFingerprint
                    model.pretrainedFingerprint = model.pretrainedFingerprint ?? model.checkpointFingerprint
                    model.checkpointFingerprint = manifest.configurationFingerprint
                    if manifest.stage == .pretraining {
                        model.pretrainedCheckpoint = manifest.id.uuidString; model.pretrainedFingerprint = manifest.configurationFingerprint
                    } else {
                        model.trainedCheckpoint = manifest.id.uuidString; model.trainedFingerprint = manifest.configurationFingerprint
                    }
                    self.store.perform { try self.store.saveModel(model) }
                }
            }
        }
    }

    func pause() { control?.set(.pause) }
    func cancel() { control?.set(.cancel) }

    func resume(model: AIModel, stage: TrainingStage) {
        guard !isBusy else { return }
        do {
            let checkpoints = CheckpointStore(root: URL(fileURLWithPath: store.preferences.checkpointsPath))
            guard let manifest = try checkpoints.latestMetadata(modelID: model.id, stage: stage) else {
                throw DataIntegrityError.invalidData("There is no saved checkpoint for this stage.")
            }
            guard manifest.epoch < manifest.settings.epochs else {
                store.notice = "This run completed all its epochs. Use \(stage == .pretraining ? "Pre-train" : "Train") to start a new run."; return
            }
            settings = manifest.settings
            start(model: model, stage: stage, resume: true)
        } catch { store.error = error.localizedDescription }
    }
}

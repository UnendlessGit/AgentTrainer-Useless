import SwiftUI

struct TrainView: View {
    var store: WorkspaceStore
    @State private var modelID: UUID?
    private var model: AIModel? { store.models.first(where: { $0.id == modelID }) }
    private var imitation: [RecordingItem] {
        guard let model else { return [] }
        return store.recordings.filter { $0.manifest.kind == .imitation && $0.eligible && (model.imitationFolderIDs.contains($0.manifest.folderID) || model.imitationRecordingIDs.contains($0.id)) }
    }
    private var pretraining: [RecordingItem] {
        guard let model else { return [] }
        return store.recordings.filter { $0.manifest.kind == .pretraining && $0.eligible && model.pretrainingFolderIDs.contains($0.manifest.folderID) }
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(title: "Train", subtitle: "Turn demonstrations into understanding, on your Mac.") { StatusPill(title: "Apple Silicon", color: .blue) }
                Surface(title: "Training model", symbol: "cpu") {
                    Picker("Model", selection: $modelID) {
                        Text("Select a model").tag(nil as UUID?)
                        ForEach(store.models) { Text($0.name).tag(Optional($0.id)) }
                    }
                    if let model { Text(model.compatibility).font(.caption).foregroundStyle(.secondary) }
                }
                HStack(alignment: .top, spacing: 20) {
                    Surface(title: "1. Pre-train", symbol: "sparkles") {
                        Text("Learn how environments change and how actions affect what happens next.").foregroundStyle(.secondary)
                        Metric(title: "ASSIGNED PRE-TRAINING RECORDINGS", value: pretraining.count.formatted())
                    }
                    Surface(title: "2. Train", symbol: "waveform.path") {
                        Text("Learn which actions to take from your selected demonstrations.").foregroundStyle(.secondary)
                        Metric(title: "ASSIGNED DEMONSTRATIONS", value: imitation.count.formatted())
                    }
                }
                EmptyState(symbol: "waveform.path", title: "Training engine in development", message: "The MLX training pipeline is being connected. This development build does not simulate training or create runnable weights.")
                    .frame(minHeight: 280)
            }.padding(30)
        }.task { modelID = store.models.first?.id }
    }
}

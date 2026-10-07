import Foundation
import Observation

enum AppTab: String, CaseIterable, Identifiable {
    case record = "Record", library = "Library", models = "AI Models", train = "Train", run = "Run", settings = "Settings"
    var id: String { rawValue }
    var symbol: String {
        switch self { case .record: "record.circle"; case .library: "square.stack.3d.up"; case .models: "cpu"; case .train: "waveform.path"; case .run: "play.circle"; case .settings: "slider.horizontal.3" }
    }
}

@MainActor @Observable
final class AppSession {
    var tab: AppTab = .record
    let store: WorkspaceStore
    let recorder: RecordingCoordinator
    init() {
        let store = WorkspaceStore()
        self.store = store
        recorder = RecordingCoordinator(store: store)
    }
}

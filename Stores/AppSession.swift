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
    let trainer: TrainingCoordinator
    let runner: RunCoordinator
    var recordingForm = RecordingForm()
    let shortcuts = GlobalShortcuts()
    private var shortcutsInstalled = false
    private var unitTestHost = false
    init() {
        var supportURL: URL?
        #if DEBUG
        if ProcessInfo.processInfo.environment["AGENTTRAINER_UNIT_TESTING"] == "1" {
            unitTestHost = true
            supportURL = FileManager.default.temporaryDirectory.appendingPathComponent("AgentTrainerUnitTests-\(UUID().uuidString)", isDirectory: true)
        } else if let path = ProcessInfo.processInfo.environment["AGENTTRAINER_VALIDATION_WORKSPACE"], path.hasPrefix("/") {
            supportURL = URL(fileURLWithPath: path, isDirectory: true)
        }
        #endif
        let store = WorkspaceStore(supportURL: supportURL)
        self.store = store
        recorder = RecordingCoordinator(store: store)
        trainer = TrainingCoordinator(store: store)
        runner = RunCoordinator(store: store)
        let formURL = store.supportURL.appendingPathComponent("recording-form.json")
        if FileManager.default.fileExists(atPath: formURL.path) {
            do { recordingForm = try AtomicFile.decode(RecordingForm.self, from: formURL) }
            catch { store.error = "Recording preferences could not be restored: \(error.localizedDescription)" }
        }
    }

    func saveRecordingForm() {
        store.perform { try store.requireWritable(); try AtomicFile.encode(recordingForm, to: store.supportURL.appendingPathComponent("recording-form.json")) }
    }

    func installShortcuts() {
        guard !unitTestHost, store.canAccessWorkspace, !shortcutsInstalled else { return }
        shortcuts.onAction = { [weak self] action in
            guard let self else { return }
            switch action {
            case .recording: Task { await self.toggleRecording(fromShortcut: true) }
            case .run:
                if self.runner.isBusy { self.runner.stop() }
                else { self.tab = .run; Task { await self.runner.start() } }
            case .emergency:
                self.runner.stop("Emergency stop.")
                Task { await self.recorder.stop(trimControlGesture: true) }
            }
        }
        do { try shortcuts.install(store.preferences.shortcuts ?? ShortcutBindings()); shortcutsInstalled = true }
        catch { store.error = error.localizedDescription }
    }

    func updateShortcuts(_ bindings: ShortcutBindings) throws {
        try store.requireWritable()
        try shortcuts.install(bindings)
        var preferences = store.preferences; preferences.shortcuts = bindings
        do { try store.savePreferences(preferences) }
        catch { try? shortcuts.install(store.preferences.shortcuts ?? ShortcutBindings()); throw error }
        shortcutsInstalled = true
    }

    func toggleRecording(fromShortcut: Bool = false) async {
        if recorder.isBusy { await recorder.stop(trimControlGesture: fromShortcut); return }
        guard !trainer.isBusy else { store.error = "Pause training before recording a new demonstration."; return }
        if recordingForm.folderID == nil { recordingForm.folderID = store.folders.first(where: { $0.kind == .imitation })?.id }
        let form = recordingForm
        var target = form.target
        if target.kind == .region {
            target.region = CaptureRect(CGRect(x: form.regionX, y: form.regionY, width: form.regionWidth, height: form.regionHeight))
            if !form.regionWindow { target.windowID = nil }
        }
        if target.kind == .display || target.kind == .desktop { target.windowID = nil }
        target.title = target.windowID != nil ? recorder.catalog.windows.first(where: { $0.id == target.windowID })?.title ?? "Window"
            : target.kind == .desktop ? "Full desktop" : recorder.catalog.displays.first(where: { $0.id == target.displayID })?.title ?? "Display"
        saveRecordingForm()
        await recorder.start(name: form.name, instruction: form.instruction, folderID: form.folderID, target: target, settings: form.settings)
    }
}

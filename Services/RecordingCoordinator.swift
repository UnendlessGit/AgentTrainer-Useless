import AppKit
import Observation

@MainActor @Observable
final class RecordingCoordinator {
    enum Phase: String { case idle = "Ready", starting = "Preparing", recording = "Recording", stopping = "Saving" }
    private(set) var phase: Phase = .idle
    private(set) var preview: NSImage?
    private(set) var manifest: RecordingManifest?
    private(set) var waitingForShortcut = false
    private var generation = UUID()
    private var clock: SessionClock?
    private var journal: RecordingJournal?
    private var input: InputCapture?
    private var visual: VisualRecorder?
    private var errorDuringSession: String?
    let store: WorkspaceStore
    let catalog = CaptureCatalog()
    let permissions = PermissionService()
    var isBusy: Bool { phase != .idle }

    init(store: WorkspaceStore) { self.store = store }

    func start(name: String, instruction: String, folderID: UUID?, target: CaptureTarget, settings: RecordingSettings,
               startingShortcut: ShortcutBinding? = nil) async {
        guard store.canAccessWorkspace, phase == .idle else { return }
        guard !store.migrating, store.activeOperations.isEmpty else { store.error = "Finish the active operation before recording."; return }
        guard let folder = store.folders.first(where: { $0.id == folderID }) else { store.error = "Choose a Library folder for this recording."; return }
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { store.error = "Give this recording a name."; return }
        permissions.refresh()
        guard permissions.screenRecording else { store.error = "Screen Recording permission is required. Enable it in Settings."; return }
        let id = UUID(); generation = id
        phase = .starting; preview = nil; manifest = nil; errorDuringSession = nil; waitingForShortcut = false
        store.activeOperations.insert("recording")
        do {
            if let shortcut = startingShortcut {
                // The control chord is not a demonstrated action. Begin the
                // session only after its keys are released, before sampling
                // initial state or creating a journal.
                let deadline = Date().addingTimeInterval(5)
                while shortcut.triggerKeyCodes.contains(where: { CGEventSource.keyState(.combinedSessionState, key: $0) }) {
                    guard generation == id, phase == .starting else { return }
                    waitingForShortcut = true
                    guard Date() < deadline else {
                        throw DataIntegrityError.invalidData("Release \(shortcut.label), then start recording again.")
                    }
                    try await Task.sleep(for: .milliseconds(10))
                }
            }
            guard generation == id, phase == .starting else { return }
            waitingForShortcut = false
            await catalog.refresh()
            guard generation == id, phase == .starting else { return }
            let (resolved, parts) = try catalog.resolve(target, settings: settings)
            guard let bounds = resolved.globalBounds?.cgRect else { throw DataIntegrityError.invalidData("No capture bounds are available.") }
            let clock = SessionClock()
            self.clock = clock
            var manifest = RecordingManifest(name: name.trimmingCharacters(in: .whitespacesAndNewlines), folderID: folder.id,
                kind: folder.kind, instruction: instruction, target: resolved, settings: settings)
            let capturesInput = settings.keyboard || settings.mouseButtons || settings.pointerMovement || settings.relativeMovement || settings.scrolling
            // Journal exists before input delivery. Input is started before visuals so initial actions are preserved.
            let journal = try RecordingJournal(root: store.recordingRoot, manifest: manifest)
            self.journal = journal
            let failure: @Sendable (String) -> Void = { [weak self] message in
                Task { @MainActor in
                    guard let self, self.generation == id else { return }
                    await self.stop(failure: message)
                }
            }
            if capturesInput {
                let input = InputCapture(clock: clock, settings: settings, shortcuts: store.preferences.shortcuts ?? ShortcutBindings(),
                                         onEvent: { try journal.append(event: $0) }, onFailure: failure)
                self.input = input
                try input.start()
                manifest.initialInputState = input.snapshot
                // Persist initial state in the journal itself, not just in the UI copy.
                try journal.setInitialInputState(manifest.initialInputState)
            }
            let visual = VisualRecorder(clock: clock, journal: journal, input: input, bounds: bounds, settings: settings,
                onPreview: { [weak self] data, snapshot in
                    Task { @MainActor in
                        guard let self, self.generation == id, self.phase == .recording else { return }
                        self.preview = NSImage(data: data); self.manifest = snapshot
                        self.store.upsertRecording(snapshot, url: journal.url)
                    }
                }, onFailure: failure)
            self.visual = visual
            try await visual.start(parts: parts)
            guard generation == id, errorDuringSession == nil, phase == .starting else { return }
            self.manifest = journal.snapshot
            store.upsertRecording(journal.snapshot, url: journal.url)
            phase = .recording
        } catch {
            guard generation == id else { return }
            await stop(failure: error.localizedDescription)
        }
    }

    func stop(failure: String? = nil, trimControlGesture: Bool = false) async {
        guard phase != .idle && phase != .stopping else { return }
        generation = UUID()
        phase = .stopping; errorDuringSession = failure; waitingForShortcut = false
        await input?.stop()
        await visual?.stop()
        if let journal, let clock {
            do {
                try journal.finish(at: clock.now, failure: failure)
                manifest = journal.snapshot
                store.upsertRecording(journal.snapshot, url: journal.url)
                if trimControlGesture, failure == nil, let boundary = input?.controlGestureBoundary,
                   boundary > 0, boundary < journal.snapshot.durationNanoseconds,
                   let item = store.recordings.first(where: { $0.id == journal.snapshot.id }) {
                    var edits = item.edits
                    edits.trimEnd = Double(boundary - 1) / 1e9
                    edits.automaticTrimReason = "Recording-control shortcut excluded. Original input events remain preserved."
                    try store.editRecording(item, edits: edits)
                }
                if failure == nil { store.notice = "Saved “\(journal.snapshot.name)” to Library." }
            } catch { store.error = "Could not finalize this recording. Its journal is recoverable: \(error.localizedDescription)" }
        }
        if let failure { store.error = failure }
        input = nil; visual = nil; journal = nil; clock = nil
        phase = .idle
        store.activeOperations.remove("recording")
    }
}

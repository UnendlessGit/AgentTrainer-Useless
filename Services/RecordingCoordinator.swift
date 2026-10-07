import AppKit
import Observation

@MainActor @Observable
final class RecordingCoordinator {
    enum Phase: String { case idle = "Ready", starting = "Preparing", recording = "Recording", stopping = "Saving" }
    private(set) var phase: Phase = .idle
    private(set) var preview: NSImage?
    private(set) var manifest: RecordingManifest?
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

    func start(name: String, instruction: String, folderID: UUID?, target: CaptureTarget, settings: RecordingSettings) async {
        guard phase == .idle else { return }
        guard !store.migrating, store.activeOperations.isEmpty else { store.error = "Finish the active operation before recording."; return }
        guard let folder = store.folders.first(where: { $0.id == folderID }) else { store.error = "Choose a Library folder for this recording."; return }
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { store.error = "Give this recording a name."; return }
        permissions.refresh()
        guard permissions.screenRecording else { store.error = "Screen Recording permission is required. Enable it in Settings."; return }
        phase = .starting; preview = nil; errorDuringSession = nil
        store.activeOperations.insert("recording")
        do {
            await catalog.refresh()
            guard phase == .starting else { return }
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
                Task { @MainActor in await self?.stop(failure: message) }
            }
            if capturesInput {
                let input = InputCapture(clock: clock, settings: settings, onEvent: { try journal.append(event: $0) }, onFailure: failure)
                self.input = input
                try input.start()
                manifest.initialInputState = input.snapshot
                // Persist initial state in the journal itself, not just in the UI copy.
                try journal.setInitialInputState(manifest.initialInputState)
            }
            let visual = VisualRecorder(clock: clock, journal: journal, input: input, bounds: bounds, settings: settings,
                onPreview: { [weak self] data, snapshot in
                    Task { @MainActor in
                        guard let self, self.phase == .recording else { return }
                        self.preview = NSImage(data: data); self.manifest = snapshot
                        self.store.upsertRecording(snapshot, url: journal.url)
                    }
                }, onFailure: failure)
            self.visual = visual
            try await visual.start(parts: parts)
            guard errorDuringSession == nil, phase == .starting else { return }
            self.manifest = journal.snapshot
            store.upsertRecording(journal.snapshot, url: journal.url)
            phase = .recording
        } catch {
            await stop(failure: error.localizedDescription)
        }
    }

    func stop(failure: String? = nil) async {
        guard phase != .idle && phase != .stopping else { return }
        phase = .stopping; errorDuringSession = failure
        await input?.stop()
        await visual?.stop()
        if let journal, let clock {
            do {
                try journal.finish(at: clock.now, failure: failure)
                manifest = journal.snapshot
                store.upsertRecording(journal.snapshot, url: journal.url)
                if failure == nil { store.notice = "Saved “\(journal.snapshot.name)” to Library." }
            } catch { store.error = "Could not finalize this recording. Its journal is recoverable: \(error.localizedDescription)" }
        }
        if let failure { store.error = failure }
        input = nil; visual = nil; journal = nil; clock = nil
        phase = .idle
        store.activeOperations.remove("recording")
    }
}

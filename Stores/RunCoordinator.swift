import AppKit
import Observation
@preconcurrency import ApplicationServices

@MainActor @Observable
final class RunCoordinator {
    enum Phase: String { case stopped = "Stopped", starting = "Preparing", running = "Running", stopping = "Stopping" }
    private(set) var phase: Phase = .stopped
    private(set) var message = "Choose a trained model and a capture target."
    private(set) var progress = RunProgress()
    private(set) var preview: NSImage?
    var configuration = RunConfiguration()
    let catalog = CaptureCatalog()
    private let store: WorkspaceStore
    private let permissions = PermissionService()
    private var executor: AgentInputExecutor?
    private var capture: CaptureFrameSource?
    private var monitor: InputCapture?
    private var watchdog: RunWatchdog?
    private var generation = UUID()
    private var workerRunning = false
    var isBusy: Bool { phase != .stopped }

    init(store: WorkspaceStore) {
        self.store = store; configuration.stopOnHumanInput = store.preferences.stopOnHumanInput
        let url = store.supportURL.appendingPathComponent("run-configuration.json")
        if let saved = try? AtomicFile.decode(RunConfiguration.self, from: url) { configuration = saved }
    }

    func saveConfiguration() {
        store.perform { try AtomicFile.encode(configuration, to: store.supportURL.appendingPathComponent("run-configuration.json")) }
    }

    func start() async {
        guard !isBusy, !store.migrating, store.activeOperations.isEmpty else { return }
        guard let model = store.models.first(where: { $0.id == configuration.modelID }), model.canRun else {
            store.error = "Select a model with a compatible imitation-learning checkpoint."; return
        }
        permissions.refresh()
        guard permissions.accessibility && permissions.screenRecording && permissions.inputMonitoring else {
            store.error = "Allow Screen Recording, Input Monitoring and Accessibility in Settings before running."; return
        }
        guard (5...1800).contains(configuration.maximumRunSeconds), (1...60).contains(configuration.maximumHoldSeconds),
              configuration.temperature.isFinite && (0.05...2).contains(configuration.temperature) else {
            store.error = "Choose valid timing and sampling limits."; return
        }
        let id = UUID(); generation = id
        phase = .starting; store.activeOperations.insert("run"); progress = RunProgress(); preview = nil
        let configuration = configuration
        let executor = AgentInputExecutor(capabilities: model.configuration.capabilities.intersecting(configuration.permissions),
                                          doubleClickInterval: NSEvent.doubleClickInterval)
        self.executor = executor
        do {
            try AtomicFile.encode(configuration, to: store.supportURL.appendingPathComponent("run-configuration.json"))
            message = "Refreshing capture sources…"
            await catalog.refresh()
            guard generation == id, !executor.isStopped else { return }
            let form = configuration.capture
            var target = form.target
            if target.kind == .region {
                target.region = CaptureRect(CGRect(x: form.regionX, y: form.regionY, width: form.regionWidth, height: form.regionHeight))
                if !form.regionWindow { target.windowID = nil }
            } else if target.kind != .window { target.windowID = nil }
            var settings = RecordingSettings(); settings.framesPerSecond = 30
            let (resolved, parts) = try catalog.resolve(target, settings: settings)
            let pid = catalog.windows.first(where: { $0.id == resolved.windowID })?.window.owningApplication?.processID
            if let pid, let window = catalog.windows.first(where: { $0.id == resolved.windowID }) { try activate(pid: pid, bounds: window.window.frame) }
            for seconds in stride(from: 3, through: 1, by: -1) {
                message = "Starting in \(seconds)… Focus the target app and release held input."
                try await Task.sleep(for: .seconds(1))
                guard generation == id, !executor.isStopped else { return }
            }
            let heldKeys = (0...127).contains { CGEventSource.keyState(.combinedSessionState, key: CGKeyCode($0)) }
            let heldButtons = (0...4).contains { CGEventSource.buttonState(.combinedSessionState, button: CGMouseButton(rawValue: UInt32($0))!) }
            guard !heldKeys && !heldButtons else { throw DataIntegrityError.invalidData("Release held keyboard keys and mouse buttons, then start the run again.") }
            let clock = SessionClock()
            let source = CaptureFrameSource(clock: clock, maximumDimension: settings.maximumDimension, onFailure: { [weak self] reason in
                executor.stop(reason)
                Task { @MainActor in self?.stop(reason) }
            })
            capture = source
            let monitor = InputCapture(clock: clock, settings: settings, onEvent: { event in
                if configuration.stopOnHumanInput { executor.stop("Stopped by keyboard or mouse input.") }
                else {
                    let owned = executor.state
                    if !owned.keys.isEmpty || !owned.buttons.isEmpty { executor.stop("Human input conflicted with input held by the agent.") }
                }
            }, onFailure: { reason in executor.stop(reason) })
            self.monitor = monitor
            try monitor.start()
            try await source.start(parts: parts)
            guard generation == id, !executor.isStopped else { await finish(id: id, reason: executor.stopReason ?? "Stopped before starting."); return }
            let request = RunRequest(model: model, configuration: configuration, preferences: store.preferences,
                target: RunTargetGuard(target: resolved, ownPID: ProcessInfo.processInfo.processIdentifier, targetPID: pid))
            watchdog = RunWatchdog(executor: executor, maximumHold: Double(configuration.maximumHoldSeconds), maximumRun: Double(configuration.maximumRunSeconds))
            phase = .running; message = "Running locally · \((store.preferences.shortcuts ?? ShortcutBindings()).emergency.label) stops immediately"
            workerRunning = true
            TrainingWorker.queue.async { [self] in
                RunWorker.run(request, source: source, clock: clock, executor: executor, humanState: { monitor.snapshot }, publish: { update, scene in
                    DispatchQueue.main.async { [self] in
                        guard generation == id else { return }
                        progress = update
                        if phase != .running { progress.heldInput = executor.state }
                        if let scene { preview = NSImage(cgImage: scene.image, size: NSSize(width: scene.image.width, height: scene.image.height)) }
                    }
                }, finished: { reason in
                    Task { @MainActor [self] in
                        guard generation == id else { return }
                        workerRunning = false
                        await finish(id: id, reason: reason)
                    }
                })
            }
        } catch {
            guard generation == id else { return }
            executor.stop(error.localizedDescription)
            await finish(id: id, reason: error.localizedDescription)
        }
    }

    /// Input release is synchronous even if the GPU is busy; stream teardown follows.
    func stop(_ reason: String = "Stopped by you.") {
        guard isBusy else { return }
        executor?.stop(reason); phase = .stopping; message = reason
        progress.heldInput = executor?.state ?? progress.heldInput
        if !workerRunning {
            let id = generation
            Task { await finish(id: id, reason: reason) }
        }
    }

    private func finish(id: UUID, reason: String) async {
        guard generation == id else { return }
        executor?.stop(reason); watchdog?.stop(); watchdog = nil
        await monitor?.stop(); await capture?.stop()
        guard generation == id else { return }
        progress.heldInput = executor?.state ?? progress.heldInput
        monitor = nil; capture = nil; executor = nil
        phase = .stopped; message = reason; store.activeOperations.remove("run")
    }

    private func activate(pid: pid_t, bounds: CGRect) throws {
        guard let application = NSRunningApplication(processIdentifier: pid) else { throw DataIntegrityError.invalidData("The target application closed.") }
        application.activate(options: [])
        let element = AXUIElementCreateApplication(pid)
        var value: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXWindowsAttribute as CFString, &value) == .success, let windows = value as? [AXUIElement] {
            for window in windows {
                var positionValue: CFTypeRef?, sizeValue: CFTypeRef?
                guard AXUIElementCopyAttributeValue(window, kAXPositionAttribute as CFString, &positionValue) == .success,
                      AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &sizeValue) == .success,
                      let positionValue, let sizeValue, CFGetTypeID(positionValue) == AXValueGetTypeID(), CFGetTypeID(sizeValue) == AXValueGetTypeID() else { continue }
                var point = CGPoint.zero, size = CGSize.zero
                AXValueGetValue(unsafeDowncast(positionValue, to: AXValue.self), .cgPoint, &point)
                AXValueGetValue(unsafeDowncast(sizeValue, to: AXValue.self), .cgSize, &size)
                if abs(point.x - bounds.minX) < 2 && abs(point.y - bounds.minY) < 2 && abs(size.width - bounds.width) < 2 {
                    AXUIElementPerformAction(window, kAXRaiseAction as CFString); break
                }
            }
        }
    }
}

import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Journal consumer of the shared capture source. Static scenes still produce
/// timed observations while referencing the last immutable image on disk.
final class VisualRecorder: @unchecked Sendable {
    private let worker = DispatchQueue(label: "com.agenttrainer.visual-journal", qos: .userInitiated)
    private let lock = NSLock()
    private var active = true
    private var timer: DispatchSourceTimer?
    private var lastImageFile: String?
    private var index: UInt64 = 0
    private var lastPreview: UInt64 = 0
    private let source: CaptureFrameSource
    private let clock: SessionClock
    private let journal: RecordingJournal
    private let images: RecordingImageStore
    private let input: InputCapture?
    private let settings: RecordingSettings
    private let onPreview: @Sendable (Data, RecordingManifest) -> Void
    private let onFailure: @Sendable (String) -> Void

    init(clock: SessionClock, journal: RecordingJournal, input: InputCapture?, bounds: CGRect,
         settings: RecordingSettings, onPreview: @escaping @Sendable (Data, RecordingManifest) -> Void,
         onFailure: @escaping @Sendable (String) -> Void) {
        self.clock = clock; self.journal = journal; self.input = input; self.settings = settings
        images = RecordingImageStore(root: journal.url)
        self.onPreview = onPreview; self.onFailure = onFailure
        source = CaptureFrameSource(clock: clock, maximumDimension: settings.maximumDimension,
                                    onDropped: { journal.noteDroppedVisualFrame() }, onFailure: onFailure)
    }

    @MainActor func start(parts: [CapturePart]) async throws {
        try await source.start(parts: parts)
        guard lock.withLock({ active }) else { throw CancellationError() }
        startSampling()
    }

    private func startSampling() {
        let timer = DispatchSource.makeTimerSource(queue: worker)
        timer.schedule(deadline: .now(), repeating: .nanoseconds(1_000_000_000 / settings.framesPerSecond), leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in self?.sample() }
        self.timer = timer; timer.resume()
    }

    @MainActor func stop() async {
        lock.withLock { active = false }
        timer?.cancel(); timer = nil
        await source.stop()
        await withCheckedContinuation { continuation in worker.async { continuation.resume() } }
    }

    private func sample() {
        guard lock.withLock({ active }) else { return }
        if input?.secureKeyboardInputActive == true {
            fail("macOS Secure Input interrupted keyboard capture. Recording stopped to avoid missing transitions."); return
        }
        autoreleasepool {
            do {
                guard let scene = try source.latestScene() else { return }
                var preview: Data?
                if !scene.reusedPixels || lastImageFile == nil {
                    let data = NSMutableData()
                    guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
                        throw DataIntegrityError.io("Could not allocate the frame encoder.")
                    }
                    CGImageDestinationAddImage(destination, scene.image, [kCGImageDestinationLossyCompressionQuality: settings.quality] as CFDictionary)
                    guard CGImageDestinationFinalize(destination) else { throw DataIntegrityError.io("The captured frame could not be encoded.") }
                    let path = try images.write(data as Data, index: index)
                    lastImageFile = path; preview = data as Data
                }
                guard let path = lastImageFile else { return }
                var state = input?.snapshot ?? InputState()
                if let cursor = CGEvent(source: nil)?.location { state.cursorX = cursor.x; state.cursorY = cursor.y }
                let available = max(clock.now, scene.sourceTime)
                try journal.append(observation: VisualObservation(id: index, timeNanoseconds: available, sourceTimeNanoseconds: scene.sourceTime,
                    imageFile: path, width: scene.image.width, height: scene.image.height, globalBounds: scene.bounds,
                    state: state, reusedPixels: scene.reusedPixels))
                index += 1
                try journal.checkpoint(at: available)
                if available >= lastPreview + 500_000_000 {
                    let data = try preview ?? Data(contentsOf: journal.url.appendingPathComponent(path))
                    onPreview(data, journal.snapshot); lastPreview = available
                }
            } catch is CancellationError { /* Expected when stopping capture. */ }
            catch { fail(error.localizedDescription) }
        }
    }

    private func fail(_ message: String) {
        let notify = lock.withLock { let old = active; active = false; return old }
        if notify { onFailure(message) }
    }
}

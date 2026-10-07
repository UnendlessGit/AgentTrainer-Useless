import Foundation
@preconcurrency import ScreenCaptureKit
import CoreImage
import Metal
import ImageIO

/// Latest-frame mailbox per source and a single sampling worker. No unbounded pixel
/// queue. Static scenes generate timed observations that reference the last image.
final class VisualRecorder: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private struct Frame {
        var buffer: CVPixelBuffer
        var time: UInt64
        var generation: UInt64
        var bounds: CGRect
    }
    private let lock = NSLock()
    private let worker = DispatchQueue(label: "com.agenttrainer.visual-journal", qos: .userInitiated)
    private let callbackQueue = DispatchQueue(label: "com.agenttrainer.capture", qos: .userInteractive)
    private var streams: [SCStream] = []
    private var streamParts: [ObjectIdentifier: CapturePart] = [:]
    private var frames: [UInt32: Frame] = [:]
    private var generation: UInt64 = 0
    private var active = true
    private var timer: DispatchSourceTimer?
    private var lastGenerations: [UInt32: UInt64] = [:]
    private var lastImageFile: String?
    private var lastSourceTime: UInt64 = 0
    private var index: UInt64 = 0
    private var lastPreview: UInt64 = 0
    private let context: CIContext
    private let clock: SessionClock
    private let journal: RecordingJournal
    private let input: InputCapture?
    private let bounds: CGRect
    private let width: Int
    private let height: Int
    private let settings: RecordingSettings
    private let onPreview: @Sendable (Data, RecordingManifest) -> Void
    private let onFailure: @Sendable (String) -> Void

    init(clock: SessionClock, journal: RecordingJournal, input: InputCapture?, bounds: CGRect,
         settings: RecordingSettings, onPreview: @escaping @Sendable (Data, RecordingManifest) -> Void,
         onFailure: @escaping @Sendable (String) -> Void) {
        self.clock = clock; self.journal = journal; self.input = input; self.bounds = bounds; self.settings = settings
        self.onPreview = onPreview; self.onFailure = onFailure
        let scale = min(1, Double(settings.maximumDimension) / max(bounds.width, bounds.height))
        width = max(2, Int((bounds.width * scale).rounded()))
        height = max(2, Int((bounds.height * scale).rounded()))
        if let device = MTLCreateSystemDefaultDevice() { context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false]) }
        else { context = CIContext(options: [.cacheIntermediates: false]) }
    }

    @MainActor func start(parts: [CapturePart]) async throws {
        do {
            for part in parts {
                let stream = SCStream(filter: part.filter, configuration: part.configuration, delegate: self)
                lock.withLock { streamParts[ObjectIdentifier(stream)] = part }
                try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: callbackQueue)
                streams.append(stream)
                try await stream.startCapture()
            }
            startSampling()
        } catch { await stop(); throw error }
    }

    // Construct DispatchSource's legacy block outside MainActor. Otherwise Swift 6
    // inherits main isolation and traps when the timer runs on the worker queue.
    private func startSampling() {
        let timer = DispatchSource.makeTimerSource(queue: worker)
        timer.schedule(deadline: .now(), repeating: .nanoseconds(1_000_000_000 / settings.framesPerSecond), leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in self?.sample() }
        self.timer = timer
        timer.resume()
    }

    @MainActor func stop() async {
        lock.withLock { active = false }
        timer?.cancel(); timer = nil
        for stream in streams { try? await stream.stopCapture() }
        streams.removeAll()
        await withCheckedContinuation { continuation in worker.async { continuation.resume() } }
        lock.withLock { frames.removeAll(); streamParts.removeAll() }
    }

    func stream(_ stream: SCStream, didStopWithError error: any Error) { fail("Screen capture stopped: \(error.localizedDescription)") }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid,
              let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = attachments.first,
              let statusCode = info[.status] as? Int, let status = SCFrameStatus(rawValue: statusCode) else { return }
        if status == .suspended || status == .stopped {
            fail("The capture target became unavailable. The recording stopped so stale images are not treated as live observations.")
            return
        }
        guard status == .complete, let buffer = sampleBuffer.imageBuffer else { return }
        let displayTicks = (info[.displayTime] as? NSNumber)?.uint64Value
        let sourceTime = displayTicks.map { clock.relative(absolute: SessionClock.nanoseconds(ticks: $0)) } ?? clock.now
        lock.withLock {
            guard active, let part = streamParts[ObjectIdentifier(stream)] else { return }
            generation += 1
            frames[part.id] = Frame(buffer: buffer, time: sourceTime, generation: generation, bounds: part.globalBounds)
        }
    }

    private func sample() {
        let snapshot = lock.withLock { (active, frames, streamParts.count) }
        guard snapshot.0 else { return }
        guard !snapshot.1.isEmpty, snapshot.1.count == snapshot.2 else {
            if clock.now > 10_000_000_000 { fail("The target did not provide a complete frame within 10 seconds.") }
            return
        }
        autoreleasepool {
            do {
                let changes = snapshot.1.mapValues(\.generation)
                let changed = changes != lastGenerations
                var preview: Data?
                if changed {
                    let rect = CGRect(x: 0, y: 0, width: width, height: height)
                    var composite = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: rect)
                    for frame in snapshot.1.values {
                        let raw = CIImage(cvPixelBuffer: frame.buffer)
                        let targetWidth = frame.bounds.width / bounds.width * Double(width)
                        let targetHeight = frame.bounds.height / bounds.height * Double(height)
                        let x = (frame.bounds.minX - bounds.minX) / bounds.width * Double(width)
                        // CG display coordinates are top-left; CI coordinates are bottom-left.
                        let y = (bounds.maxY - frame.bounds.maxY) / bounds.height * Double(height)
                        let transformed = raw.transformed(by: CGAffineTransform(scaleX: targetWidth / raw.extent.width, y: targetHeight / raw.extent.height))
                            .transformed(by: CGAffineTransform(translationX: x, y: y))
                        composite = transformed.composited(over: composite)
                    }
                    guard let data = context.jpegRepresentation(of: composite, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                        options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: settings.quality]) else {
                        throw DataIntegrityError.io("The captured frame could not be encoded.")
                    }
                    let path = String(format: "frames/%012llu.jpg", index)
                    try AtomicFile.write(data, to: journal.url.appendingPathComponent(path))
                    lastImageFile = path
                    lastSourceTime = snapshot.1.values.map(\.time).max() ?? 0
                    lastGenerations = changes
                    preview = data
                }
                guard let path = lastImageFile else { return }
                // VisualObservation availability is stamped AFTER preprocessing and image persistence.
                // Events during this work remain aligned to the previous observation.
                let state = input?.snapshot ?? InputState()
                let available = max(clock.now, lastSourceTime)
                try journal.append(observation: VisualObservation(id: index, timeNanoseconds: available, sourceTimeNanoseconds: lastSourceTime,
                    imageFile: path, width: width, height: height, globalBounds: CaptureRect(bounds), state: state, reusedPixels: !changed))
                index += 1
                try journal.checkpoint(at: available)
                if available >= lastPreview + 500_000_000 {
                    let data = try preview ?? Data(contentsOf: journal.url.appendingPathComponent(path))
                    onPreview(data, journal.snapshot)
                    lastPreview = available
                }
            } catch { fail(error.localizedDescription) }
        }
    }

    private func fail(_ message: String) {
        let notify = lock.withLock { let old = active; active = false; return old }
        if notify { onFailure(message) }
    }
}

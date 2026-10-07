import Foundation
@preconcurrency import ScreenCaptureKit
import CoreImage
import Metal

struct CapturedScene: @unchecked Sendable {
    let image: CGImage
    let bounds: CaptureRect
    let sourceTime: UInt64
    let availableTime: UInt64
    let generation: UInt64
    let reusedPixels: Bool
}

/// One bounded ScreenCaptureKit mailbox and renderer shared by recording and run.
/// Rendering is called from exactly one consumer worker, never from the UI thread.
final class CaptureFrameSource: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private struct Frame {
        var buffer: CVPixelBuffer
        var time: UInt64
        var generation: UInt64
        var bounds: CGRect
    }
    private let lock = NSLock()
    private let callbackQueue = DispatchQueue(label: "com.agenttrainer.capture", qos: .userInteractive)
    private var streams: [SCStream] = []
    private var parts: [ObjectIdentifier: CapturePart] = [:]
    private var frames: [UInt32: Frame] = [:]
    private var consumed: [UInt32: UInt64] = [:]
    private var generation: UInt64 = 0
    private var active = true
    private let clock: SessionClock
    private let maximumDimension: Int
    private let onFailure: @Sendable (String) -> Void
    private let onDropped: @Sendable () -> Void
    private let context: CIContext
    // These caches are confined to the consumer worker.
    private var renderedGenerations: [UInt32: UInt64] = [:]
    private var renderedScene: CapturedScene?

    init(clock: SessionClock, maximumDimension: Int, onDropped: @escaping @Sendable () -> Void = {},
         onFailure: @escaping @Sendable (String) -> Void) {
        self.clock = clock; self.maximumDimension = maximumDimension
        self.onDropped = onDropped; self.onFailure = onFailure
        if let device = MTLCreateSystemDefaultDevice() { context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false]) }
        else { context = CIContext(options: [.cacheIntermediates: false]) }
    }

    @MainActor func start(parts: [CapturePart]) async throws {
        do {
            for part in parts {
                guard lock.withLock({ active }) else { throw CancellationError() }
                let stream = SCStream(filter: part.filter, configuration: part.configuration, delegate: self)
                lock.withLock { self.parts[ObjectIdentifier(stream)] = part }
                try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: callbackQueue)
                streams.append(stream)
                try await stream.startCapture()
                guard lock.withLock({ active }) else { throw CancellationError() }
            }
        } catch { await stop(); throw error }
    }

    @MainActor func stop() async {
        lock.withLock { active = false }
        for stream in streams { try? await stream.stopCapture() }
        streams.removeAll()
        lock.withLock { frames.removeAll(); parts.removeAll() }
    }

    func latestScene() throws -> CapturedScene? {
        let snapshot = lock.withLock { () -> (Bool, [UInt32: Frame], Int) in
            if frames.count == parts.count { consumed = frames.mapValues(\.generation) }
            return (active, frames, parts.count)
        }
        guard snapshot.0 else { throw CancellationError() }
        guard !snapshot.1.isEmpty, snapshot.1.count == snapshot.2 else {
            if clock.now > 10_000_000_000 { throw DataIntegrityError.io("The target did not provide a complete frame within 10 seconds.") }
            return nil
        }
        let generations = snapshot.1.mapValues(\.generation)
        if generations == renderedGenerations, let scene = renderedScene {
            return CapturedScene(image: scene.image, bounds: scene.bounds, sourceTime: scene.sourceTime,
                availableTime: clock.now, generation: scene.generation, reusedPixels: true)
        }
        let bounds = snapshot.1.values.reduce(CGRect.null) { $0.union($1.bounds) }
        let scale = min(1, Double(maximumDimension) / max(bounds.width, bounds.height))
        let width = max(2, Int((bounds.width * scale).rounded())), height = max(2, Int((bounds.height * scale).rounded()))
        let rectangle = CGRect(x: 0, y: 0, width: width, height: height)
        var composite = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: rectangle)
        for frame in snapshot.1.values {
            let raw = CIImage(cvPixelBuffer: frame.buffer)
            let targetWidth = frame.bounds.width / bounds.width * Double(width)
            let targetHeight = frame.bounds.height / bounds.height * Double(height)
            let x = (frame.bounds.minX - bounds.minX) / bounds.width * Double(width)
            let y = (bounds.maxY - frame.bounds.maxY) / bounds.height * Double(height)
            let transformed = raw.transformed(by: CGAffineTransform(scaleX: targetWidth / raw.extent.width, y: targetHeight / raw.extent.height))
                .transformed(by: CGAffineTransform(translationX: x, y: y))
            composite = transformed.composited(over: composite)
        }
        guard let image = context.createCGImage(composite, from: rectangle, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else {
            throw DataIntegrityError.io("Could not render the captured observation.")
        }
        let sourceTime = snapshot.1.values.map(\.time).max() ?? 0
        let scene = CapturedScene(image: image, bounds: CaptureRect(bounds), sourceTime: sourceTime,
            availableTime: max(clock.now, sourceTime), generation: snapshot.1.values.map(\.generation).max() ?? 0, reusedPixels: false)
        renderedScene = scene; renderedGenerations = generations
        return scene
    }

    func stream(_ stream: SCStream, didStopWithError error: any Error) { fail("Screen capture stopped: \(error.localizedDescription)") }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid,
              let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = attachments.first, let code = info[.status] as? Int, let status = SCFrameStatus(rawValue: code) else { return }
        if status == .suspended || status == .stopped { fail("The capture target became unavailable. Stale pixels cannot be used as live observations."); return }
        guard status == .complete, let buffer = sampleBuffer.imageBuffer else { return }
        let displayTicks = (info[.displayTime] as? NSNumber)?.uint64Value
        let sourceTime = displayTicks.map { clock.relative(absolute: SessionClock.nanoseconds(ticks: $0)) } ?? clock.now
        var dropped = false
        var invalidCrop = false
        lock.withLock {
            guard active, let part = parts[ObjectIdentifier(stream)] else { return }
            if let old = frames[part.id], old.generation > (consumed[part.id] ?? 0) { dropped = true }
            generation += 1
            var bounds = part.globalBounds
            if part.tracksWindowGeometry, let dictionary = info[.screenRect] as? NSDictionary,
               let screenRect = CGRect(dictionaryRepresentation: dictionary as CFDictionary), screenRect.width > 0, screenRect.height > 0 {
                bounds = screenRect
                if let crop = part.crop {
                    guard CGRect(origin: .zero, size: screenRect.size).contains(crop) else { invalidCrop = true; return }
                    bounds = crop.offsetBy(dx: screenRect.minX, dy: screenRect.minY)
                }
            }
            frames[part.id] = Frame(buffer: buffer, time: sourceTime, generation: generation, bounds: bounds)
        }
        if invalidCrop { fail("The selected region no longer fits inside its window. Resize the region before recording again."); return }
        if dropped { onDropped() }
    }

    private func fail(_ message: String) {
        let notify = lock.withLock { let wasActive = active; active = false; return wasActive }
        if notify { onFailure(message) }
    }
}

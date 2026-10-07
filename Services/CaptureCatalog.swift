import AppKit
@preconcurrency import ScreenCaptureKit
import Observation

struct DisplayChoice: Identifiable {
    let display: SCDisplay
    let title: String
    var id: UInt32 { display.displayID }
}
struct WindowChoice: Identifiable {
    let window: SCWindow
    let title: String
    var id: UInt32 { window.windowID }
}

struct CapturePart: @unchecked Sendable {
    let id: UInt32
    let filter: SCContentFilter
    let configuration: SCStreamConfiguration
    let globalBounds: CGRect
}

@MainActor @Observable
final class CaptureCatalog {
    private(set) var displays: [DisplayChoice] = []
    private(set) var windows: [WindowChoice] = []
    private(set) var loading = false
    var error: String?

    func refresh() async {
        guard !loading else { return }
        loading = true
        defer { loading = false }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            displays = content.displays.enumerated().map { offset, display in
                let name = NSScreen.screens.first(where: { ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? UInt32) == display.displayID })?.localizedName
                return DisplayChoice(display: display, title: "\(name ?? "Display \(offset + 1)") · \(display.width) × \(display.height)")
            }
            windows = content.windows.filter {
                $0.windowLayer == 0 && $0.frame.width > 50 && $0.frame.height > 50
                    && $0.owningApplication?.processID != ProcessInfo.processInfo.processIdentifier
            }.map { WindowChoice(window: $0, title: "\($0.owningApplication?.applicationName ?? "Application") — \($0.title?.isEmpty == false ? $0.title! : "Untitled window")") }
                .sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
            error = nil
        } catch { self.error = error.localizedDescription }
    }

    func resolve(_ target: CaptureTarget, settings: RecordingSettings) throws -> (CaptureTarget, [CapturePart]) {
        var parts: [(UInt32, SCContentFilter, CGRect, CGRect?)] = []
        switch (target.kind, target.windowID) {
        case (.desktop, _):
            guard !displays.isEmpty else { throw DataIntegrityError.invalidData("No displays are available. Refresh capture sources.") }
            for choice in displays {
                parts.append((choice.id, SCContentFilter(display: choice.display, excludingWindows: []), choice.display.frame, nil))
            }
        case (.display, _), (.region, nil):
            guard let choice = displays.first(where: { $0.id == target.displayID }) else { throw DataIntegrityError.invalidData("Choose an available display.") }
            var bounds = choice.display.frame
            var crop: CGRect?
            if target.kind == .region {
                guard let region = target.region, region.isValid,
                      CGRect(origin: .zero, size: bounds.size).contains(region.cgRect) else { throw DataIntegrityError.invalidData("The region must fit within the selected display.") }
                crop = region.cgRect
                bounds = region.cgRect.offsetBy(dx: bounds.minX, dy: bounds.minY)
            }
            parts.append((choice.id, SCContentFilter(display: choice.display, excludingWindows: []), bounds, crop))
        case (.window, _), (.region, .some):
            guard let choice = windows.first(where: { $0.id == target.windowID }) else { throw DataIntegrityError.invalidData("Choose an available window.") }
            var bounds = choice.window.frame
            var crop: CGRect?
            if target.kind == .region {
                guard let region = target.region, region.isValid,
                      CGRect(origin: .zero, size: bounds.size).contains(region.cgRect) else { throw DataIntegrityError.invalidData("The region must fit within the selected window.") }
                crop = region.cgRect
                bounds = region.cgRect.offsetBy(dx: bounds.minX, dy: bounds.minY)
            }
            parts.append((choice.id, SCContentFilter(desktopIndependentWindow: choice.window), bounds, crop))
        }
        let union = parts.reduce(CGRect.null) { $0.union($1.2) }
        let scale = min(1, Double(settings.maximumDimension) / max(union.width, union.height))
        var resolved = target
        resolved.globalBounds = CaptureRect(union)
        let result = parts.map { id, filter, bounds, crop in
            let config = SCStreamConfiguration()
            // ScreenCaptureKit's underlying surface path requires even dimensions
            // for some window sizes, even when requesting BGRA output.
            config.width = max(2, Int((bounds.width * scale / 2).rounded()) * 2)
            config.height = max(2, Int((bounds.height * scale / 2).rounded()) * 2)
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(settings.framesPerSecond))
            config.queueDepth = 3
            config.showsCursor = settings.includesCursor
            config.capturesAudio = false
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.colorSpaceName = CGColorSpace.sRGB
            config.ignoreShadowsSingleWindow = true
            if let crop { config.sourceRect = crop }
            return CapturePart(id: id, filter: filter, configuration: config, globalBounds: bounds)
        }
        return (resolved, result)
    }
}

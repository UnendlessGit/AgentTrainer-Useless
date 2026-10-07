import Foundation

struct CaptureRect: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
    init(_ rect: CGRect) { x = rect.minX; y = rect.minY; width = rect.width; height = rect.height }
    var isValid: Bool { [x, y, width, height].allSatisfy(\.isFinite) && width > 0 && height > 0 }
}

enum CaptureKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case desktop = "Full desktop", display = "Display", window = "Window", region = "Region"
    var id: String { rawValue }
    var symbol: String {
        switch self { case .desktop: "macwindow.on.rectangle"; case .display: "display"; case .window: "macwindow"; case .region: "viewfinder" }
    }
}

struct CaptureTarget: Codable, Equatable, Sendable {
    var kind: CaptureKind = .display
    var displayID: UInt32?
    var windowID: UInt32?
    var region: CaptureRect?
    var title = "Choose a display"
    var globalBounds: CaptureRect?
}

struct RecordingSettings: Codable, Equatable, Sendable {
    var maximumDimension = 1280
    var framesPerSecond = 15
    var quality = 0.85
    var includesCursor = true
    var keyboard = true
    var mouseButtons = true
    var pointerMovement = true
    var scrolling = true
    var relativeMovement = false
}

enum LibraryKind: String, Codable, CaseIterable, Identifiable, Sendable {
    case imitation = "Demonstrations", pretraining = "Pre-training"
    var id: String { rawValue }
}

struct LibraryFolder: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var name: String
    var kind: LibraryKind
    var parentID: UUID?
    var createdAt = Date()
}

enum RecordingStatus: String, Codable, Sendable {
    case recording, complete, interrupted, failed
}

struct RecordingManifest: Codable, Identifiable, Sendable {
    var schemaVersion = 1
    var actionSchemaVersion = ComputerAction.schemaVersion
    var id = UUID()
    var name: String
    var folderID: UUID
    var kind: LibraryKind
    var createdAt = Date()
    var instruction = ""
    var target: CaptureTarget
    var settings: RecordingSettings
    var status: RecordingStatus = .recording
    var durationNanoseconds: UInt64 = 0
    var observationCount = 0
    var inputEventCount = 0
    var droppedVisualFrames = 0
    var initialInputState = InputState()
    var failure: String?
    var duration: Double { Double(durationNanoseconds) / 1_000_000_000 }
    var eligibility: String {
        if let failure { return failure }
        if status == .recording { return "Recording" }
        if status != .complete { return "Review recovered data" }
        if observationCount < 2 { return "Needs more observations" }
        if kind == .imitation && inputEventCount == 0 { return "Wait-only demonstration" }
        return "Ready for training"
    }
}

struct VisualObservation: Codable, Identifiable, Equatable, Sendable {
    var id: UInt64
    /// Logical observation availability. Never earlier than the captured pixels.
    var timeNanoseconds: UInt64
    /// Host display time of the original pixels; unchanged when a static frame is reused.
    var sourceTimeNanoseconds: UInt64
    var imageFile: String
    var width: Int
    var height: Int
    var globalBounds: CaptureRect
    var state: InputState
    var reusedPixels: Bool
}

struct RecordingEdits: Codable, Equatable, Sendable {
    var schemaVersion = 1
    var name: String?
    var instruction: String?
    var trimStart: Double = 0
    var trimEnd: Double?
    var excluded = false
    var automaticTrimReason: String?
    var reviewedRecovery: Bool?
}

struct RecordingItem: Identifiable, Sendable {
    var manifest: RecordingManifest
    var edits: RecordingEdits
    var url: URL
    var id: UUID { manifest.id }
    var name: String { edits.name ?? manifest.name }
    var instruction: String { edits.instruction ?? manifest.instruction }
    var duration: Double { max(0, min(edits.trimEnd ?? manifest.duration, manifest.duration) - edits.trimStart) }
    var eligible: Bool {
        let approved = manifest.status == .complete && manifest.failure == nil
            || manifest.status == .interrupted && edits.reviewedRecovery == true
        return !edits.excluded && approved && manifest.observationCount >= 2 && duration > 0
    }
    var eligibility: String {
        if edits.excluded { return "Excluded" }
        if eligible && manifest.status == .interrupted { return "Reviewed recovery" }
        return manifest.eligibility
    }
}

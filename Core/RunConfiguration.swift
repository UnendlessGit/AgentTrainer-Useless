import Foundation

struct RunConfiguration: Codable, Equatable, Sendable {
    var modelID: UUID?
    var capture = RecordingForm()
    var instruction = ""
    var permissions = ActionCapabilities()
    var stopOnHumanInput = true
    var deterministic = true
    var temperature: Float = 0.8
    var maximumRunSeconds = 60
    var maximumHoldSeconds = 10
    var useBestCheckpoint = true
}

struct PolicyDecision: Sendable {
    var action: ComputerAction
    var delay: Double
    var inferenceSeconds: Double
    var observationBounds: CaptureRect
}

extension ActionCapabilities {
    func intersecting(_ other: ActionCapabilities) -> ActionCapabilities {
        var result = self
        result.keys.formIntersection(other.keys); result.buttons.formIntersection(other.buttons)
        result.pointer = pointer && other.pointer; result.relativePointer = relativePointer && other.relativePointer
        result.scrolling = scrolling && other.scrolling; result.dragging = dragging && other.dragging
        result.chords = chords && other.chords; result.maximumHeldKeys = min(maximumHeldKeys, other.maximumHeldKeys)
        result.repeatsKeys = repeatsKeys && other.repeatsKeys
        return result
    }
}

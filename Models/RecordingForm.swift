import Foundation

struct RecordingForm: Codable, Equatable, Sendable {
    var name = "Untitled demonstration"
    var instruction = ""
    var folderID: UUID?
    var target = CaptureTarget()
    var settings = RecordingSettings()
    var regionX = 0.0
    var regionY = 0.0
    var regionWidth = 800.0
    var regionHeight = 600.0
    var regionWindow = false
}

import AppKit
import CoreGraphics
@preconcurrency import ApplicationServices
import Observation

@MainActor @Observable
final class PermissionService {
    private(set) var screenRecording = false
    private(set) var inputMonitoring = false
    private(set) var accessibility = false

    func refresh() {
        screenRecording = CGPreflightScreenCaptureAccess()
        inputMonitoring = CGPreflightListenEventAccess()
        accessibility = AXIsProcessTrusted()
    }
    func requestScreenRecording() { CGRequestScreenCaptureAccess(); refresh() }
    func requestInputMonitoring() { CGRequestListenEventAccess(); refresh() }
    func requestAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
        refresh()
    }
    func openPrivacy(_ pane: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)") { NSWorkspace.shared.open(url) }
    }
}

import Foundation
import CoreGraphics
@preconcurrency import ApplicationServices

struct RunTargetGuard: Sendable {
    let target: CaptureTarget
    let ownPID: pid_t
    let targetPID: pid_t?

    func validate(observedBounds: CaptureRect, action: ComputerAction? = nil, state: InputState = InputState()) throws {
        guard AXIsProcessTrusted(), CGPreflightScreenCaptureAccess(), CGPreflightListenEventAccess() else {
            throw DataIntegrityError.io("A required capture or input permission was revoked. The run stopped.")
        }
        let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        guard let foreground = focusedApplicationPID(), foreground != ownPID else {
            throw DataIntegrityError.invalidData("Focus the application you want the agent to control.")
        }
        if let windowID = target.windowID {
            guard let window = windows.first(where: { ($0[kCGWindowNumber as String] as? NSNumber)?.uint32Value == windowID }),
                  let bounds = rectangle(window), let pid = (window[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value,
                  pid == targetPID, foreground == pid else {
                throw DataIntegrityError.invalidData("The selected window is hidden, closed or no longer focused. The run stopped.")
            }
            // Window-list order includes non-activating auxiliary windows (for
            // example TextEdit's writing controls). Accessibility identifies the
            // real keyboard recipient; z-order is checked separately for pointers.
            guard let focused = focusedWindowBounds(pid: pid), nearlyEqual(focused, bounds) else {
                throw DataIntegrityError.invalidData("Another window or dialog has keyboard focus. The run stopped.")
            }
            if let point = Self.pointerDestination(action, state: state) {
                for candidate in windows {
                    if (candidate[kCGWindowNumber as String] as? NSNumber)?.uint32Value == windowID { break }
                    guard ((candidate[kCGWindowAlpha as String] as? NSNumber)?.doubleValue ?? 1) > 0.01,
                          let candidateBounds = rectangle(candidate), candidateBounds.contains(point) else { continue }
                    throw DataIntegrityError.invalidData("Another window covers the pointer destination. The run stopped.")
                }
            }
            if let crop = target.region, target.kind == .region,
               !CGRect(origin: .zero, size: bounds.size).contains(crop.cgRect) {
                throw DataIntegrityError.invalidData("The selected region no longer fits inside its window.")
            }
            let current = target.kind == .region && target.region != nil ? target.region!.cgRect.offsetBy(dx: bounds.minX, dy: bounds.minY) : bounds
            guard abs(current.minX - observedBounds.x) < 2, abs(current.minY - observedBounds.y) < 2,
                  abs(current.width - observedBounds.width) < 2, abs(current.height - observedBounds.height) < 2 else {
                throw DataIntegrityError.invalidData("The selected window moved or resized during a decision. Restart with its new position.")
            }
        } else if let displayID = target.displayID, target.kind != .desktop {
            guard CGDisplayIsActive(displayID) != 0 else { throw DataIntegrityError.invalidData("The selected display disconnected.") }
            if let window = windows.first(where: {
                ($0[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == foreground && ($0[kCGWindowLayer as String] as? NSNumber)?.intValue == 0
            }), let bounds = rectangle(window) {
                guard bounds.intersects(observedBounds.cgRect) else { throw DataIntegrityError.invalidData("The focused application is outside the selected capture target.") }
            } else { throw DataIntegrityError.invalidData("No focused application window is available inside the selected capture target.") }
        }
    }

    private func rectangle(_ window: [String: Any]) -> CGRect? {
        guard let dictionary = window[kCGWindowBounds as String] as? NSDictionary else { return nil }
        return CGRect(dictionaryRepresentation: dictionary as CFDictionary)
    }

    static func pointerDestination(_ action: ComputerAction?, state: InputState) -> CGPoint? {
        switch action {
        case .pointer(let x, let y): return CGPoint(x: x, y: y)
        case .relativePointer(let dx, let dy): return CGPoint(x: state.cursorX + dx, y: state.cursorY + dy)
        case .buttonDown, .scroll: return CGPoint(x: state.cursorX, y: state.cursorY)
        default: return nil
        }
    }

    private func nearlyEqual(_ a: CGRect, _ b: CGRect) -> Bool {
        abs(a.minX - b.minX) < 2 && abs(a.minY - b.minY) < 2 && abs(a.width - b.width) < 2 && abs(a.height - b.height) < 2
    }

    private func focusedWindowBounds(pid: pid_t) -> CGRect? {
        let application = AXUIElementCreateApplication(pid)
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(application, kAXFocusedWindowAttribute as CFString, &focusedValue) == .success,
              let focusedValue, CFGetTypeID(focusedValue) == AXUIElementGetTypeID() else { return nil }
        let window = unsafeDowncast(focusedValue, to: AXUIElement.self)
        var children: CFTypeRef?
        if AXUIElementCopyAttributeValue(window, kAXChildrenAttribute as CFString, &children) == .success,
           let children = children as? [AXUIElement] {
            for child in children {
                var role: CFTypeRef?
                if AXUIElementCopyAttributeValue(child, kAXRoleAttribute as CFString, &role) == .success,
                   role as? String == kAXSheetRole { return nil }
            }
        }
        var positionValue: CFTypeRef?, sizeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(window, kAXPositionAttribute as CFString, &positionValue) == .success,
              AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &sizeValue) == .success,
              let positionValue, let sizeValue, CFGetTypeID(positionValue) == AXValueGetTypeID(), CFGetTypeID(sizeValue) == AXValueGetTypeID() else { return nil }
        var position = CGPoint.zero, size = CGSize.zero
        guard AXValueGetValue(unsafeDowncast(positionValue, to: AXValue.self), .cgPoint, &position),
              AXValueGetValue(unsafeDowncast(sizeValue, to: AXValue.self), .cgSize, &size) else { return nil }
        return CGRect(origin: position, size: size)
    }

    private func focusedApplicationPID() -> pid_t? {
        let system = AXUIElementCreateSystemWide()
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(system, kAXFocusedApplicationAttribute as CFString, &value) == .success, let value,
              CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        var pid: pid_t = 0
        AXUIElementGetPid(unsafeDowncast(value, to: AXUIElement.self), &pid)
        return pid
    }
}

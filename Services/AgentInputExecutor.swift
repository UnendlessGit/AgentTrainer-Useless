import Foundation
import CoreGraphics
import Carbon

/// Owns only agent-held inputs. Cancellation and emission share a lock, so no
/// press can race past an emergency release. Waiting and ML work never hold it.
final class AgentInputExecutor: @unchecked Sendable {
    static let marker: Int64 = 0x4154524E
    private let lock = NSLock()
    private var owned = InputState()
    private var stopped = false
    private var reason: String?
    private var lastClick: (button: Int, time: TimeInterval, point: CGPoint, count: Int)?
    private var holds: [String: TimeInterval] = [:]
    private let capabilities: ActionCapabilities
    private let eventSource = CGEventSource(stateID: .privateState)
    private let doubleClickInterval: TimeInterval
    private let emit: @Sendable (CGEvent) -> Void

    init(capabilities: ActionCapabilities, doubleClickInterval: TimeInterval = 0.5,
         emit: @escaping @Sendable (CGEvent) -> Void = { $0.post(tap: .cghidEventTap) }) {
        self.capabilities = capabilities; self.doubleClickInterval = doubleClickInterval; self.emit = emit
        if let position = CGEvent(source: nil)?.location { owned.cursorX = position.x; owned.cursorY = position.y }
    }

    var isStopped: Bool { lock.withLock { stopped } }
    var stopReason: String? { lock.withLock { reason } }
    var longestHold: TimeInterval { lock.withLock { holds.values.min().map { ProcessInfo.processInfo.systemUptime - $0 } ?? 0 } }
    var state: InputState {
        lock.withLock {
            var value = owned
            if let cursor = CGEvent(source: nil)?.location { value.cursorX = cursor.x; value.cursorY = cursor.y }
            return value
        }
    }

    func stop(_ reason: String) {
        lock.withLock {
            guard !stopped else { return }
            stopped = true; self.reason = reason
            for key in owned.keys.sorted() { _ = send(.keyUp(code: key)) }
            for button in owned.buttons.sorted() { _ = send(.buttonUp(button: button)) }
        }
    }

    func execute(_ action: ComputerAction, bounds: CaptureRect) throws {
        try lock.withLock {
            guard !stopped else { throw CancellationError() }
            guard !IsSecureEventInputEnabled() else { throw DataIntegrityError.io("Secure Input became active. The run stopped.") }
            if let cursor = CGEvent(source: nil)?.location { owned.cursorX = cursor.x; owned.cursorY = cursor.y }
            guard capabilities.permits(action, state: owned) else { throw DataIntegrityError.invalidData("The requested action violates the current input permissions or held state.") }
            guard bounds.isValid else { throw DataIntegrityError.invalidData("The capture target has invalid bounds.") }
            switch action {
            case .pointer(let x, let y):
                guard bounds.cgRect.contains(CGPoint(x: x, y: y)) else { throw DataIntegrityError.invalidData("Pointer action is outside the selected capture target.") }
            case .relativePointer(let dx, let dy):
                guard abs(dx) <= bounds.width, abs(dy) <= bounds.height,
                      bounds.cgRect.contains(CGPoint(x: owned.cursorX + dx, y: owned.cursorY + dy)) else {
                    throw DataIntegrityError.invalidData("Relative movement would leave the selected capture target.")
                }
            case .scroll(let dx, let dy):
                guard abs(dx) <= 256, abs(dy) <= 256, bounds.cgRect.contains(CGPoint(x: owned.cursorX, y: owned.cursorY)) else {
                    throw DataIntegrityError.invalidData("Scrolling is outside the permitted range or capture target.")
                }
            case .buttonDown:
                guard bounds.cgRect.contains(CGPoint(x: owned.cursorX, y: owned.cursorY)) else {
                    throw DataIntegrityError.invalidData("The pointer is outside the selected capture target. Move it inside before running.")
                }
            default: break
            }
            guard send(action) else { throw DataIntegrityError.io("macOS could not allocate an input event. The run stopped.") }
        }
    }

    private func send(_ action: ComputerAction) -> Bool {
        let position = CGPoint(x: owned.cursorX, y: owned.cursorY)
        var event: CGEvent?
        switch action {
        case .keyDown(let code), .keyUp(let code):
            let down: Bool
            if case .keyDown = action { down = true } else { down = false }
            event = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: down)
            if (54...63).contains(code) { event?.type = .flagsChanged }
        case .buttonDown(let button), .buttonUp(let button):
            let down: Bool
            if case .buttonDown = action { down = true } else { down = false }
            let type: CGEventType = button == 0 ? (down ? .leftMouseDown : .leftMouseUp)
                : button == 1 ? (down ? .rightMouseDown : .rightMouseUp) : (down ? .otherMouseDown : .otherMouseUp)
            event = CGEvent(mouseEventSource: eventSource, mouseType: type, mouseCursorPosition: position, mouseButton: CGMouseButton(rawValue: UInt32(button))!)
            if down {
                let now = ProcessInfo.processInfo.systemUptime
                let previous = lastClick
                let continues = previous.map { $0.button == button && now - $0.time <= doubleClickInterval && hypot($0.point.x - position.x, $0.point.y - position.y) < 4 } ?? false
                lastClick = (button, now, position, continues ? min(3, (previous?.count ?? 0) + 1) : 1)
            }
            event?.setIntegerValueField(.mouseEventClickState, value: Int64(lastClick?.count ?? 1))
        case .pointer(let x, let y): event = movement(to: CGPoint(x: x, y: y))
        case .relativePointer(let dx, let dy):
            event = movement(to: CGPoint(x: position.x + dx, y: position.y + dy))
            event?.setIntegerValueField(.mouseEventDeltaX, value: Int64(dx.rounded()))
            event?.setIntegerValueField(.mouseEventDeltaY, value: Int64(dy.rounded()))
        case .scroll(let dx, let dy):
            event = CGEvent(scrollWheelEvent2Source: eventSource, units: .pixel, wheelCount: 2,
                wheel1: Int32(clamping: Int(dy.rounded())), wheel2: Int32(clamping: Int(dx.rounded())), wheel3: 0)
            event?.location = position
            event?.setDoubleValueField(.scrollWheelEventPointDeltaAxis1, value: dy)
            event?.setDoubleValueField(.scrollWheelEventPointDeltaAxis2, value: dx)
        case .wait: return true
        }
        guard let event else { return false }
        owned.apply(action)
        switch action {
        case .keyDown(let key): holds["key\(key)"] = ProcessInfo.processInfo.systemUptime
        case .keyUp(let key): holds.removeValue(forKey: "key\(key)")
        case .buttonDown(let button): holds["button\(button)"] = ProcessInfo.processInfo.systemUptime
        case .buttonUp(let button): holds.removeValue(forKey: "button\(button)")
        default: break
        }
        event.flags = Self.flags(for: owned.keys)
        event.setIntegerValueField(.eventSourceUserData, value: Self.marker)
        emit(event)
        return true
    }

    private func movement(to point: CGPoint) -> CGEvent? {
        let button = owned.buttons.sorted().first
        let type: CGEventType = button == 0 ? .leftMouseDragged : button == 1 ? .rightMouseDragged : button != nil ? .otherMouseDragged : .mouseMoved
        return CGEvent(mouseEventSource: eventSource, mouseType: type, mouseCursorPosition: point,
                       mouseButton: CGMouseButton(rawValue: UInt32(button ?? 0))!)
    }

    static func flags(for keys: Set<UInt16>) -> CGEventFlags {
        var raw: UInt64 = 0
        let mappings: [(UInt16, UInt64, UInt64)] = [(55, 1 << 20, 0x8), (54, 1 << 20, 0x10), (56, 1 << 17, 0x2),
            (60, 1 << 17, 0x4), (59, 1 << 18, 0x1), (62, 1 << 18, 0x2000), (58, 1 << 19, 0x20), (61, 1 << 19, 0x40),
            (57, 1 << 16, 0), (63, 1 << 23, 0)]
        for (key, aggregate, side) in mappings where keys.contains(key) { raw |= aggregate | side }
        return CGEventFlags(rawValue: raw)
    }
}

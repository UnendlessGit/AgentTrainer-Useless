import Foundation
import CoreGraphics
import Carbon

/// The event tap callback does no disk I/O. A bounded queue writes every accepted
/// transition; overflow stops recording visibly instead of silently losing input.
final class InputCapture: @unchecked Sendable {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "com.agenttrainer.input-journal", qos: .userInteractive)
    private let slots = DispatchSemaphore(value: 16_384)
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var state = InputState()
    private var active = false
    private var controlGesture: ControlGestureTracker
    private let clock: SessionClock
    private let settings: RecordingSettings
    private let onEvent: @Sendable (InputTransition) throws -> Void
    private let onFailure: @Sendable (String) -> Void

    init(clock: SessionClock, settings: RecordingSettings, shortcuts: ShortcutBindings = ShortcutBindings(),
         onEvent: @escaping @Sendable (InputTransition) throws -> Void,
         onFailure: @escaping @Sendable (String) -> Void) {
        self.clock = clock; self.settings = settings; self.onEvent = onEvent; self.onFailure = onFailure
        controlGesture = ControlGestureTracker(bindings: [shortcuts.recording, shortcuts.emergency])
    }

    var snapshot: InputState { lock.withLock { state } }
    var controlGestureBoundary: UInt64? { lock.withLock { controlGesture.boundary } }
    var secureKeyboardInputActive: Bool { settings.keyboard && IsSecureEventInputEnabled() }

    @MainActor func start() throws {
        guard !secureKeyboardInputActive else { throw DataIntegrityError.io("macOS Secure Input is active. Close the password field or secure-input application before recording keyboard actions.") }
        var initial = InputState()
        if settings.keyboard {
            initial.keys = Set((0...127).compactMap { CGEventSource.keyState(.combinedSessionState, key: CGKeyCode($0)) ? UInt16($0) : nil })
        }
        if settings.mouseButtons {
            initial.buttons = Set((0...4).filter { CGEventSource.buttonState(.combinedSessionState, button: CGMouseButton(rawValue: UInt32($0))!) })
        }
        let cursor = CGEvent(source: nil)?.location ?? .zero
        initial.cursorX = cursor.x; initial.cursorY = cursor.y
        lock.withLock { state = initial; active = true }
        let types: [CGEventType] = [.keyDown, .keyUp, .flagsChanged, .leftMouseDown, .leftMouseUp, .rightMouseDown,
                                   .rightMouseUp, .otherMouseDown, .otherMouseUp, .mouseMoved, .leftMouseDragged,
                                   .rightMouseDragged, .otherMouseDragged, .scrollWheel]
        let mask = types.reduce(CGEventMask(0)) { $0 | (CGEventMask(1) << $1.rawValue) }
        guard let eventTap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
            options: .listenOnly, eventsOfInterest: mask, callback: { _, type, event, context in
                guard let context else { return Unmanaged.passUnretained(event) }
                let owner = Unmanaged<InputCapture>.fromOpaque(context).takeUnretainedValue()
                owner.receive(type: type, event: event)
                return Unmanaged.passUnretained(event)
            }, userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
                lock.withLock { active = false }
                throw DataIntegrityError.io("Input Monitoring permission is required to record actions. Enable it in Settings, then try again.")
            }
        tap = eventTap
        source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: eventTap, enable: true)
    }

    @MainActor func stop() async {
        lock.withLock { active = false }
        if let tap { CGEvent.tapEnable(tap: tap, enable: false); CFMachPortInvalidate(tap) }
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        tap = nil; source = nil
        await withCheckedContinuation { continuation in queue.async { continuation.resume() } }
    }

    private func receive(type: CGEventType, event: CGEvent) {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            fail("Input monitoring was interrupted. The recording stopped to avoid silently missing actions.")
            return
        }
        guard lock.withLock({ active }) else { return }
        // Synthetic policy events carry a marker and cannot contaminate human demonstrations.
        guard event.getIntegerValueField(.eventSourceUserData) != 0x4154524E else { return }
        let code = UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode))
        var action: ComputerAction?
        switch type {
        case .keyDown where settings.keyboard: action = .keyDown(code: code)
        case .keyUp where settings.keyboard: action = .keyUp(code: code)
        case .flagsChanged where settings.keyboard:
            guard let down = ModifierState.isDown(code: code, flags: event.flags.rawValue) else { break }
            let wasDown = lock.withLock { state.keys.contains(code) }
            if down != wasDown { action = down ? .keyDown(code: code) : .keyUp(code: code) }
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            if settings.mouseButtons { action = .buttonDown(button: Int(event.getIntegerValueField(.mouseEventButtonNumber))) }
        case .leftMouseUp, .rightMouseUp, .otherMouseUp:
            if settings.mouseButtons { action = .buttonUp(button: Int(event.getIntegerValueField(.mouseEventButtonNumber))) }
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            if settings.relativeMovement {
                action = .relativePointer(dx: Double(event.getIntegerValueField(.mouseEventDeltaX)), dy: Double(event.getIntegerValueField(.mouseEventDeltaY)))
            } else if settings.pointerMovement { action = .pointer(x: event.location.x, y: event.location.y) }
        case .scrollWheel where settings.scrolling:
            action = .scroll(dx: event.getDoubleValueField(.scrollWheelEventPointDeltaAxis2), dy: event.getDoubleValueField(.scrollWheelEventPointDeltaAxis1))
        default: break
        }
        guard let action else { return }
        let time = clock.relative(absolute: event.timestamp)
        lock.withLock {
            controlGesture.observe(action: action, flags: event.flags.rawValue, time: time)
            state.apply(action)
            state.cursorX = event.location.x; state.cursorY = event.location.y
        }
        let transition = InputTransition(id: 0, timeNanoseconds: time, action: action,
                                         isRepeat: type == .keyDown && event.getIntegerValueField(.keyboardEventAutorepeat) != 0,
                                         modifiers: event.flags.rawValue, cursorX: event.location.x, cursorY: event.location.y,
                                         rawDeltaX: event.getIntegerValueField(.mouseEventDeltaX),
                                         rawDeltaY: event.getIntegerValueField(.mouseEventDeltaY))
        guard slots.wait(timeout: .now()) == .success else {
            fail("The input journal could not keep up. Recording stopped before dropping input transitions.")
            return
        }
        queue.async { [self] in
            defer { slots.signal() }
            do { try onEvent(transition) } catch { fail(error.localizedDescription) }
        }
    }

    private func fail(_ message: String) {
        let shouldNotify = lock.withLock { let old = active; active = false; return old }
        if shouldNotify { onFailure(message) }
    }
}

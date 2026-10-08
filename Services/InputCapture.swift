import Foundation
import CoreGraphics
import Carbon
import AppKit

/// macOS volume keys arrive as NX_SYSDEFINED / AUX_CONTROL_BUTTONS, not
/// ordinary virtual-key events. Keep the existing key vocabulary while using
/// the native event payload in both directions (IOKit's IOLLEvent/ev_keymap).
enum MediaKeyEvent {
    static let type = CGEventType(rawValue: 14)!
    static let flavors: [UInt16: Int] = [72: 0, 73: 1, 74: 7]

    struct Transition: Equatable {
        var code: UInt16
        var down: Bool
        var isRepeat: Bool
    }

    static func decode(_ event: CGEvent) -> Transition? {
        guard event.type == type, let native = NSEvent(cgEvent: event), native.subtype.rawValue == 8,
              let code = flavors.first(where: { $0.value == (native.data1 >> 16) & 0xffff })?.key else { return nil }
        let state = (native.data1 >> 8) & 0xff
        guard state == 10 || state == 11 else { return nil }
        return Transition(code: code, down: state == 10, isRepeat: state == 10 && native.data1 & 1 != 0)
    }

    static func make(code: UInt16, down: Bool, isRepeat: Bool, source: CGEventSource?) -> CGEvent? {
        guard let flavor = flavors[code] else { return nil }
        let payload = (flavor << 16) | ((down ? 10 : 11) << 8) | (down && isRepeat ? 1 : 0)
        let event = NSEvent.otherEvent(with: .systemDefined, location: .zero, modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: 0, context: nil,
            subtype: 8, data1: payload, data2: -1)?.cgEvent
        event?.setSource(source)
        return event
    }
}

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
    private let onInput: @Sendable (InputState) -> Void
    private let onEvent: @Sendable (InputTransition) throws -> Void
    private let onFailure: @Sendable (String) -> Void

    init(clock: SessionClock, settings: RecordingSettings, shortcuts: ShortcutBindings = ShortcutBindings(),
         onInput: @escaping @Sendable (InputState) -> Void = { _ in },
         onEvent: @escaping @Sendable (InputTransition) throws -> Void,
         onFailure: @escaping @Sendable (String) -> Void) {
        self.clock = clock; self.settings = settings; self.onInput = onInput; self.onEvent = onEvent; self.onFailure = onFailure
        controlGesture = ControlGestureTracker(bindings: [shortcuts.recording, shortcuts.emergency])
    }

    var snapshot: InputState { lock.withLock { state } }
    var controlGestureBoundary: UInt64? { lock.withLock { controlGesture.stopBoundary } }
    var secureKeyboardInputActive: Bool { settings.keyboard && IsSecureEventInputEnabled() }

    @MainActor func start() throws {
        guard !secureKeyboardInputActive else { throw DataIntegrityError.io("macOS Secure Input is active. Close the password field or secure-input application before recording keyboard actions.") }
        var initial = InputState()
        if settings.keyboard {
            initial.keys = Set((0...127).compactMap { CGEventSource.keyState(.combinedSessionState, key: CGKeyCode($0)) ? UInt16($0) : nil })
        }
        if settings.mouseButtons {
            initial.buttons = Set(ActionCapabilities.supportedButtons.filter { CGEventSource.buttonState(.combinedSessionState, button: CGMouseButton(rawValue: UInt32($0))!) })
        }
        let cursor = CGEvent(source: nil)?.location ?? .zero
        initial.cursorX = cursor.x; initial.cursorY = cursor.y
        lock.withLock { state = initial; active = true }
        let types: [CGEventType] = [.keyDown, .keyUp, .flagsChanged, .leftMouseDown, .leftMouseUp, .rightMouseDown,
                                   .rightMouseUp, .otherMouseDown, .otherMouseUp, .mouseMoved, .leftMouseDragged,
                                   .rightMouseDragged, .otherMouseDragged, .scrollWheel, MediaKeyEvent.type]
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
        var isRepeat = type == .keyDown && event.getIntegerValueField(.keyboardEventAutorepeat) != 0
        switch type {
        case MediaKeyEvent.type where settings.keyboard:
            if let media = MediaKeyEvent.decode(event) {
                action = media.down ? .keyDown(code: media.code) : .keyUp(code: media.code)
                isRepeat = media.isRepeat
            }
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
        let updatedState = lock.withLock {
            controlGesture.observe(action: action, flags: event.flags.rawValue, time: time)
            state.apply(action)
            // Auxiliary key payloads have no meaningful pointer location.
            if type != MediaKeyEvent.type {
                state.cursorX = event.location.x; state.cursorY = event.location.y
            }
            return state
        }
        onInput(updatedState)
        let transition = InputTransition(id: 0, timeNanoseconds: time, action: action,
                                         isRepeat: isRepeat,
                                         modifiers: event.flags.rawValue,
                                         cursorX: type == MediaKeyEvent.type ? nil : event.location.x,
                                         cursorY: type == MediaKeyEvent.type ? nil : event.location.y,
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

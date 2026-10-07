import Foundation

/// Versioned semantics shared by recording, dataset alignment, training and execution.
/// A transition changes one part of state; chords and drags are ordered transitions,
/// never independent predictions of conflicting held-state snapshots.
enum ComputerAction: Codable, Equatable, Sendable {
    case keyDown(code: UInt16)
    case keyUp(code: UInt16)
    case pointer(x: Double, y: Double)
    case relativePointer(dx: Double, dy: Double)
    case buttonDown(button: Int)
    case buttonUp(button: Int)
    case scroll(dx: Double, dy: Double)
    case wait(seconds: Double)

    static let schemaVersion = 1

    var label: String {
        switch self {
        case .keyDown(let code): return "↓ \(KeyNames.name(code))"
        case .keyUp(let code): return "↑ \(KeyNames.name(code))"
        case .pointer(let x, let y): return String(format: "Pointer %.1f, %.1f", x, y)
        case .relativePointer(let dx, let dy): return String(format: "Relative %+.1f, %+.1f", dx, dy)
        case .buttonDown(let button): return "↓ \(KeyNames.button(button))"
        case .buttonUp(let button): return "↑ \(KeyNames.button(button))"
        case .scroll(let dx, let dy): return String(format: "Scroll %+.1f, %+.1f", dx, dy)
        case .wait(let seconds): return String(format: "Wait %.3f s", seconds)
        }
    }
}

struct InputState: Codable, Equatable, Sendable {
    var keys: Set<UInt16> = []
    var buttons: Set<Int> = []
    var cursorX: Double = 0
    var cursorY: Double = 0

    mutating func apply(_ action: ComputerAction) {
        switch action {
        case .keyDown(let code): keys.insert(code)
        case .keyUp(let code): keys.remove(code)
        case .buttonDown(let button): buttons.insert(button)
        case .buttonUp(let button): buttons.remove(button)
        case .pointer(let x, let y): cursorX = x; cursorY = y
        case .relativePointer(let dx, let dy): cursorX += dx; cursorY += dy
        case .scroll, .wait: break
        }
    }
}

struct ActionCapabilities: Codable, Equatable, Sendable {
    var keys: Set<UInt16> = Set(KeyNames.names.keys)
    var pointer = true
    var relativePointer = false
    var buttons: Set<Int> = [0, 1]
    var scrolling = true
    var dragging = true
    var chords = true
    var maximumHeldKeys = 6

    /// Apply before sampling AND execution. Releases for already-held input always
    /// remain valid so changing permissions cannot strand a key or button.
    func permits(_ action: ComputerAction, state: InputState) -> Bool {
        switch action {
        case .keyDown(let code):
            return keys.contains(code) && !state.keys.contains(code)
                && state.keys.count < maximumHeldKeys && (chords || state.keys.isEmpty)
        case .keyUp(let code): return state.keys.contains(code)
        case .buttonDown(let button): return buttons.contains(button) && !state.buttons.contains(button)
        case .buttonUp(let button): return state.buttons.contains(button)
        case .pointer(let x, let y):
            return pointer && x.isFinite && y.isFinite && (dragging || state.buttons.isEmpty)
        case .relativePointer(let dx, let dy):
            return relativePointer && dx.isFinite && dy.isFinite && (dragging || state.buttons.isEmpty)
        case .scroll(let dx, let dy): return scrolling && dx.isFinite && dy.isFinite
        case .wait(let seconds): return seconds.isFinite && seconds >= 0 && seconds <= 10
        }
    }
}

struct InputTransition: Codable, Equatable, Identifiable, Sendable {
    var id: UInt64
    /// Nanoseconds since session origin, from CGEvent's host-clock timestamp.
    var timeNanoseconds: UInt64
    var action: ComputerAction
    var isRepeat = false
    var modifiers: UInt64 = 0
    /// Preserve pointer location on clicks and scrolls even when move logging is off.
    var cursorX: Double? = nil
    var cursorY: Double? = nil
    var rawDeltaX: Int64? = nil
    var rawDeltaY: Int64? = nil
}

enum KeyNames {
    static let names: [UInt16: String] = [
        0: "A", 1: "S", 2: "D", 3: "F", 4: "H", 5: "G", 6: "Z", 7: "X", 8: "C", 9: "V",
        11: "B", 12: "Q", 13: "W", 14: "E", 15: "R", 16: "Y", 17: "T", 18: "1", 19: "2",
        20: "3", 21: "4", 22: "6", 23: "5", 24: "=", 25: "9", 26: "7", 27: "−", 28: "8",
        29: "0", 30: "]", 31: "O", 32: "U", 33: "[", 34: "I", 35: "P", 36: "Return", 37: "L",
        38: "J", 39: "'", 40: "K", 41: ";", 42: "\\", 43: ",", 44: "/", 45: "N", 46: "M",
        47: ".", 48: "Tab", 49: "Space", 50: "`", 51: "Delete", 53: "Escape", 54: "Right Command",
        55: "Command", 56: "Shift", 57: "Caps Lock", 58: "Option", 59: "Control", 60: "Right Shift",
        61: "Right Option", 62: "Right Control", 63: "Fn", 123: "←", 124: "→", 125: "↓", 126: "↑",
        10: "ISO Section", 64: "F17", 65: "Keypad Decimal", 67: "Keypad Multiply", 69: "Keypad Plus",
        71: "Keypad Clear", 72: "Volume Up", 73: "Volume Down", 74: "Mute", 75: "Keypad Divide",
        76: "Keypad Enter", 78: "Keypad Minus", 79: "F18", 80: "F19", 81: "Keypad Equals",
        82: "Keypad 0", 83: "Keypad 1", 84: "Keypad 2", 85: "Keypad 3", 86: "Keypad 4", 87: "Keypad 5",
        88: "Keypad 6", 89: "Keypad 7", 90: "F20", 91: "Keypad 8", 92: "Keypad 9", 93: "JIS Yen",
        94: "JIS Underscore", 95: "Keypad Comma", 96: "F5", 97: "F6", 98: "F7", 99: "F3", 100: "F8",
        101: "F9", 102: "JIS Eisu", 103: "F11", 104: "JIS Kana", 105: "F13", 106: "F16", 107: "F14",
        109: "F10", 110: "Context Menu", 111: "F12", 113: "F15", 114: "Help", 115: "Home",
        116: "Page Up", 117: "Forward Delete", 118: "F4", 119: "End", 120: "F2", 121: "Page Down", 122: "F1"
    ]
    static func name(_ code: UInt16) -> String { names[code] ?? "Key \(code)" }
    static func button(_ button: Int) -> String {
        switch button { case 0: "Left button"; case 1: "Right button"; case 2: "Middle button"; default: "Button \(button)" }
    }
}

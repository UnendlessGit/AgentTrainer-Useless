import Foundation

/// Detect only the contiguous modifier prefix of a configured control shortcut.
/// An intervening task action ends the prefix, so genuine earlier work is never
/// trimmed just because a modifier was held for a long time.
struct ControlGestureTracker {
    var bindings: [ShortcutBinding]
    private var modifierPrefix: UInt64?
    private(set) var boundary: UInt64?
    // Carbon can deliver the stop command before the event tap receives the
    // shortcut's final key. Its observed modifier prefix still needs exclusion.
    var stopBoundary: UInt64? { boundary ?? modifierPrefix }

    mutating func observe(action: ComputerAction, flags: UInt64, time: UInt64) {
        if case .keyDown(let code) = action, bindings.contains(where: { binding in
            guard binding.keyCode == code else { return false }
            let mask: UInt64 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20)
            let required: UInt64 = (binding.shift ? 1 << 17 : 0) | (binding.control ? 1 << 18 : 0)
                | (binding.option ? 1 << 19 : 0) | (binding.command ? 1 << 20 : 0)
            return flags & mask == required
        }) {
            boundary = modifierPrefix ?? time
            modifierPrefix = nil
            return
        }
        if case .keyDown(let code) = action, [54, 55, 56, 58, 59, 60, 61, 62].contains(code) {
            if modifierPrefix == nil { modifierPrefix = time }
        } else { modifierPrefix = nil }
    }
}

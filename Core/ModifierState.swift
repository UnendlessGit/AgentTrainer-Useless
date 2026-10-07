import Foundation

enum ModifierState {
    /// Device-specific flags preserve independent left/right modifier transitions.
    /// Querying current global keyboard state can race a short press or a queued
    /// event and does not reflect synthetic events used for repeatable validation.
    static func isDown(code: UInt16, flags: UInt64) -> Bool? {
        let masks: (side: UInt64, pair: UInt64, aggregate: UInt64)
        switch code {
        case 56: masks = (0x2, 0x6, 1 << 17)
        case 60: masks = (0x4, 0x6, 1 << 17)
        case 59: masks = (0x1, 0x2001, 1 << 18)
        case 62: masks = (0x2000, 0x2001, 1 << 18)
        case 58: masks = (0x20, 0x60, 1 << 19)
        case 61: masks = (0x40, 0x60, 1 << 19)
        case 55: masks = (0x8, 0x18, 1 << 20)
        case 54: masks = (0x10, 0x18, 1 << 20)
        case 57: return flags & (1 << 16) != 0
        case 63: return flags & (1 << 23) != 0
        default: return nil
        }
        if flags & masks.pair != 0 { return flags & masks.side != 0 }
        return flags & masks.aggregate != 0
    }
}

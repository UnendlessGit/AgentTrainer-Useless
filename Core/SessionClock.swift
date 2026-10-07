import Foundation
import Darwin

struct SessionClock: Sendable {
    let origin: UInt64
    static let timebase: mach_timebase_info_data_t = {
        var value = mach_timebase_info_data_t()
        mach_timebase_info(&value)
        return value
    }()
    init() { origin = Self.absoluteNanoseconds() }
    static func nanoseconds(ticks: UInt64) -> UInt64 {
        // Divide before multiplying to avoid overflow on long-running machines.
        let divisor = UInt64(timebase.denom), multiplier = UInt64(timebase.numer)
        return (ticks / divisor) * multiplier + (ticks % divisor) * multiplier / divisor
    }
    static func absoluteNanoseconds() -> UInt64 { nanoseconds(ticks: mach_absolute_time()) }
    func relative(absolute: UInt64) -> UInt64 { absolute > origin ? absolute - origin : 0 }
    var now: UInt64 { relative(absolute: Self.absoluteNanoseconds()) }
}

import XCTest
@testable import AgentTrainer

final class ControlGestureTests: XCTestCase {
    func testOnlyContiguousShortcutPrefixIsExcluded() {
        let flags: UInt64 = (1 << 17) | (1 << 20)
        var tracker = ControlGestureTracker(bindings: [ShortcutBindings().recording])
        tracker.observe(action: .keyDown(code: 55), flags: 1 << 20, time: 100)
        tracker.observe(action: .keyDown(code: 56), flags: flags, time: 110)
        XCTAssertEqual(tracker.stopBoundary, 100, "The prefix remains available if Carbon handles stop before the final key arrives.")
        tracker.observe(action: .keyDown(code: 15), flags: flags, time: 120)
        tracker.observe(action: .keyUp(code: 15), flags: flags, time: 130)
        XCTAssertEqual(tracker.boundary, 100)

        tracker = ControlGestureTracker(bindings: [ShortcutBindings().recording])
        tracker.observe(action: .keyDown(code: 55), flags: 1 << 20, time: 100)
        tracker.observe(action: .keyDown(code: 0), flags: 1 << 20, time: 110)
        tracker.observe(action: .keyUp(code: 0), flags: 1 << 20, time: 115)
        XCTAssertNil(tracker.stopBoundary, "Task input ends a candidate control prefix.")
        tracker.observe(action: .keyDown(code: 15), flags: flags, time: 120)
        XCTAssertEqual(tracker.boundary, 120)
    }
}

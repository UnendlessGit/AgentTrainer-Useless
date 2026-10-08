import XCTest
@testable import AgentTrainer

final class ControlGestureTests: XCTestCase {
    func testStartShortcutReleaseIncludesEitherModifierSideButNotUnrelatedKeys() {
        let original = ShortcutBindings().recording
        XCTAssertTrue(Set<UInt16>([15, 54, 55, 58, 59, 61, 62]).isSubset(of: original.triggerKeyCodes))
        XCTAssertTrue(original.triggerKeyCodes.isDisjoint(with: [0, 56, 57, 60, 63]))
        let custom = ShortcutBinding(keyCode: 17, command: false, shift: true, option: false, control: false)
        XCTAssertEqual(custom.triggerKeyCodes, [17, 56, 60])
    }

    func testOnlyContiguousShortcutPrefixIsExcluded() {
        let flags: UInt64 = (1 << 17) | (1 << 20)
        var tracker = ControlGestureTracker(bindings: [ShortcutBinding(keyCode: 15)])
        tracker.observe(action: .keyDown(code: 55), flags: 1 << 20, time: 100)
        tracker.observe(action: .keyDown(code: 56), flags: flags, time: 110)
        XCTAssertEqual(tracker.stopBoundary, 100, "The prefix remains available if Carbon handles stop before the final key arrives.")
        tracker.observe(action: .keyDown(code: 15), flags: flags, time: 120)
        tracker.observe(action: .keyUp(code: 15), flags: flags, time: 130)
        XCTAssertEqual(tracker.boundary, 100)

        tracker = ControlGestureTracker(bindings: [ShortcutBinding(keyCode: 15)])
        tracker.observe(action: .keyDown(code: 55), flags: 1 << 20, time: 100)
        tracker.observe(action: .keyDown(code: 0), flags: 1 << 20, time: 110)
        tracker.observe(action: .keyUp(code: 0), flags: 1 << 20, time: 115)
        XCTAssertNil(tracker.stopBoundary, "Task input ends a candidate control prefix.")
        tracker.observe(action: .keyDown(code: 15), flags: flags, time: 120)
        XCTAssertEqual(tracker.boundary, 120)
    }

    func testDefaultThreeModifierStopRetainsTheFullPrefixBoundary() {
        let flags: UInt64 = (1 << 18) | (1 << 19) | (1 << 20)
        var tracker = ControlGestureTracker(bindings: [ShortcutBindings().recording])
        tracker.observe(action: .keyDown(code: 55), flags: 1 << 20, time: 100)
        tracker.observe(action: .keyDown(code: 58), flags: (1 << 19) | (1 << 20), time: 110)
        tracker.observe(action: .keyDown(code: 59), flags: flags, time: 120)
        tracker.observe(action: .keyDown(code: 15), flags: flags, time: 130)
        XCTAssertEqual(tracker.boundary, 100)
    }

    func testUnreachableOrConflictingEmergencyBindingsAreRejected() throws {
        try ShortcutBindings().validate()
        for code: UInt32 in [55, 57, 63, 72, 74, UInt32.max] {
            var bindings = ShortcutBindings(); bindings.emergency.keyCode = code
            XCTAssertThrowsError(try bindings.validate())
            XCTAssertFalse(bindings.emergency.label.isEmpty, "Malformed saved keys must not crash presentation.")
        }
        var bindings = ShortcutBindings(); bindings.emergency = bindings.recording
        XCTAssertThrowsError(try bindings.validate())
        bindings.emergency = ShortcutBinding(keyCode: 53, command: false, shift: false, option: false, control: false)
        XCTAssertThrowsError(try bindings.validate())
    }
}

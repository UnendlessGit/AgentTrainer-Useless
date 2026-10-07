import XCTest
@testable import AgentTrainer

final class ActionGrammarTests: XCTestCase {
    func testLeftAndRightModifierTransitionsUseTheEventNotLaterGlobalState() {
        XCTAssertEqual(ModifierState.isDown(code: 56, flags: 0x20002), true)
        XCTAssertEqual(ModifierState.isDown(code: 60, flags: 0x20006), true)
        XCTAssertEqual(ModifierState.isDown(code: 56, flags: 0x20004), false)
        XCTAssertEqual(ModifierState.isDown(code: 60, flags: 0), false)
        XCTAssertEqual(ModifierState.isDown(code: 55, flags: 0x100008), true)
        XCTAssertEqual(ModifierState.isDown(code: 55, flags: 0x100010), false)
        XCTAssertNil(ModifierState.isDown(code: 0, flags: 0))
    }
    func testIllegalTransitionsAndChordsAreMasked() {
        var capability = ActionCapabilities()
        capability.chords = false
        var state = InputState()
        XCTAssertFalse(capability.permits(.keyUp(code: 0), state: state))
        XCTAssertTrue(capability.permits(.keyDown(code: 0), state: state))
        state.apply(.keyDown(code: 0))
        XCTAssertFalse(capability.permits(.keyDown(code: 0), state: state))
        XCTAssertFalse(capability.permits(.keyDown(code: 1), state: state))
        XCTAssertTrue(capability.permits(.keyUp(code: 0), state: state))
    }

    func testRevokedPermissionsCannotStrandHeldInputs() {
        var state = InputState()
        state.apply(.keyDown(code: 0)); state.apply(.buttonDown(button: 0))
        var capability = ActionCapabilities()
        capability.keys = []; capability.buttons = []
        XCTAssertTrue(capability.permits(.keyUp(code: 0), state: state))
        XCTAssertTrue(capability.permits(.buttonUp(button: 0), state: state))
        XCTAssertFalse(capability.permits(.keyDown(code: 0), state: InputState()))
    }

    func testDragRestrictionsAndFiniteCoordinates() {
        var capability = ActionCapabilities(); capability.dragging = false
        var state = InputState(); state.apply(.buttonDown(button: 0))
        XCTAssertFalse(capability.permits(.pointer(x: 10, y: 10), state: state))
        XCTAssertFalse(capability.permits(.pointer(x: .nan, y: 0), state: InputState()))
        XCTAssertFalse(capability.permits(.scroll(dx: .infinity, dy: 0), state: InputState()))
        XCTAssertFalse(capability.permits(.wait(seconds: -1), state: InputState()))
    }

    func testFingerprintIsStableAcrossSetInsertionOrderAndChangesWithSemantics() {
        var a = PolicyConfiguration(), b = PolicyConfiguration()
        a.capabilities.keys = Set([0, 1, 2]); b.capabilities.keys = Set([2, 0, 1])
        XCTAssertEqual(a.fingerprint, b.fingerprint)
        b.capabilities.dragging = false
        XCTAssertNotEqual(a.fingerprint, b.fingerprint)
    }
}

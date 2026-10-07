import XCTest
import CoreGraphics
@testable import AgentTrainer

final class DisplayLayoutConstraintTests: XCTestCase {
    func testSelectedDisplayRejectsGeometryChangesButAllowsUnrelatedDisplays() throws {
        let bounds = CGRect(x: -1600, y: 0, width: 1600, height: 1000)
        let selected = DisplayLayoutConstraint(expected: [7: bounds], includesEntireDesktop: false)
        XCTAssertNoThrow(try selected.validate(current: [7: bounds, 8: CGRect(x: 0, y: 0, width: 1920, height: 1080)]))
        XCTAssertThrowsError(try selected.validate(current: [7: bounds.offsetBy(dx: 100, dy: 0)]))
        XCTAssertThrowsError(try selected.validate(current: [7: CGRect(x: -1600, y: 0, width: 1280, height: 800)]))
        XCTAssertThrowsError(try selected.validate(current: [:]))
    }

    func testDesktopRejectsAddedOrRemovedDisplayEvenIfUnionIsUnchanged() throws {
        let first = CGRect(x: 0, y: 0, width: 1920, height: 1080)
        let second = CGRect(x: 1920, y: 0, width: 1600, height: 1000)
        let desktop = DisplayLayoutConstraint(expected: [7: first, 8: second], includesEntireDesktop: true)
        XCTAssertNoThrow(try desktop.validate(current: [7: first, 8: second]))
        XCTAssertThrowsError(try desktop.validate(current: [7: first]))
        XCTAssertThrowsError(try desktop.validate(current: [7: first, 8: second, 9: first]))
    }
}

import XCTest
import CoreGraphics
@testable import AgentTrainer

final class RunTargetGuardTests: XCTestCase {
    func testCursorDoesNotBlockClickButCoveringPanelDoes() {
        let point = CGPoint(x: 288, y: 348)
        func window(_ id: UInt32, _ level: Int32, _ rect: CGRect, alpha: Double = 1) -> [String: Any] {
            [kCGWindowNumber as String: NSNumber(value: id), kCGWindowLayer as String: NSNumber(value: level),
             kCGWindowAlpha as String: NSNumber(value: alpha), kCGWindowBounds as String: rect.dictionaryRepresentation]
        }
        let cursor = window(6, CGWindowLevelForKey(.cursorWindow), CGRect(x: 282, y: 342, width: 28, height: 40))
        let target = window(7836, 0, CGRect(x: 200, y: 190, width: 230, height: 408))
        let panel = window(8, CGWindowLevelForKey(.floatingWindow), CGRect(x: 270, y: 330, width: 100, height: 100))
        let invisible = window(9, 0, CGRect(x: 0, y: 0, width: 1000, height: 1000), alpha: 0)
        XCTAssertNil(RunTargetGuard.pointerObstruction(in: [cursor, invisible, target, panel], targetWindowID: 7836, point: point))
        let obstruction = RunTargetGuard.pointerObstruction(in: [cursor, panel, target], targetWindowID: 7836, point: point)
        XCTAssertEqual((obstruction?[kCGWindowNumber as String] as? NSNumber)?.uint32Value, 8)
        XCTAssertNil(RunTargetGuard.pointerObstruction(in: [cursor, panel, target], targetWindowID: 7836, point: CGPoint(x: 210, y: 400)))
    }
}

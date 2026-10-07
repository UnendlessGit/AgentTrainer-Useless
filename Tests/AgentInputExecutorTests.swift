import XCTest
import CoreGraphics
@testable import AgentTrainer

private final class CapturedInputEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [(CGEventType, Int64, UInt64, Int64, Int64)] = []
    func append(_ event: CGEvent) {
        lock.withLock { events.append((event.type, event.getIntegerValueField(.keyboardEventKeycode), event.flags.rawValue, event.getIntegerValueField(.eventSourceUserData), event.getIntegerValueField(.keyboardEventAutorepeat))) }
    }
    var values: [(CGEventType, Int64, UInt64, Int64, Int64)] { lock.withLock { events } }
}

final class AgentInputExecutorTests: XCTestCase {
    func testExplicitRepeatRequiresOwnedKeyAndDoesNotRestartHoldDeadline() throws {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        let bounds = CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100))
        XCTAssertThrowsError(try executor.execute(.keyRepeat(code: 51), bounds: bounds))
        try executor.execute(.keyDown(code: 51), bounds: bounds)
        Thread.sleep(forTimeInterval: 0.01)
        let holdBeforeRepeat = executor.longestHold
        try executor.execute(.keyRepeat(code: 51), bounds: bounds)
        XCTAssertGreaterThanOrEqual(executor.longestHold, holdBeforeRepeat)
        XCTAssertEqual(executor.state.keys, [51])
        executor.stop("Emergency")
        XCTAssertEqual(events.values.map { $0.0 }, [.keyDown, .keyDown, .keyUp])
        XCTAssertEqual(events.values.map { $0.4 }, [0, 1, 0])
        XCTAssertTrue(executor.state.keys.isEmpty)
    }

    func testEmergencyReleasesOwnedChordAndPreventsFurtherPresses() throws {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        let bounds = CaptureRect(CGRect(x: -10_000, y: -10_000, width: 20_000, height: 20_000))
        try executor.execute(.keyDown(code: 55), bounds: bounds)
        try executor.execute(.keyDown(code: 0), bounds: bounds)
        XCTAssertEqual(executor.state.keys, [0, 55])
        executor.stop("Emergency")
        XCTAssertTrue(executor.state.keys.isEmpty)
        XCTAssertTrue(executor.isStopped)
        XCTAssertEqual(events.values.count, 4)
        XCTAssertEqual(events.values[1].2 & (1 << 20), 1 << 20)
        XCTAssertEqual(events.values.last?.2, 0)
        XCTAssertTrue(events.values.allSatisfy { $0.3 == AgentInputExecutor.marker })
        XCTAssertThrowsError(try executor.execute(.keyDown(code: 1), bounds: bounds))
        executor.stop("Again")
        XCTAssertEqual(events.values.count, 4)
    }

    func testRejectsOutOfBoundsInvalidTransitionsAndUnownedReleases() throws {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        let bounds = CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100))
        XCTAssertThrowsError(try executor.execute(.keyUp(code: 0), bounds: bounds))
        XCTAssertThrowsError(try executor.execute(.pointer(x: 101, y: 10), bounds: bounds))
        XCTAssertThrowsError(try executor.execute(.scroll(dx: .greatestFiniteMagnitude, dy: 0), bounds: bounds))
        XCTAssertThrowsError(try executor.execute(.wait(seconds: -.infinity), bounds: bounds))
        XCTAssertTrue(events.values.isEmpty)
    }

    func testStopWinsAgainstConcurrentActionEmission() {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        let bounds = CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100))
        let group = DispatchGroup()
        for key: UInt16 in [0, 1] {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                for _ in 0..<100 {
                    do { try executor.execute(.keyDown(code: key), bounds: bounds); try executor.execute(.keyUp(code: key), bounds: bounds) }
                    catch { break }
                }
            }
        }
        executor.stop("Emergency")
        XCTAssertEqual(group.wait(timeout: .now() + 2), .success)
        XCTAssertTrue(executor.state.keys.isEmpty)
        let count = events.values.count
        XCTAssertThrowsError(try executor.execute(.keyDown(code: 0), bounds: bounds))
        XCTAssertEqual(events.values.count, count)
    }
}

import XCTest
import CoreGraphics
import AppKit
@testable import AgentTrainer

private final class CapturedInputEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [(CGEventType, Int64, UInt64, Int64, Int64)] = []
    private var media: [MediaKeyEvent.Transition] = []
    func append(_ event: CGEvent) {
        lock.withLock {
            events.append((event.type, event.getIntegerValueField(.keyboardEventKeycode), event.flags.rawValue, event.getIntegerValueField(.eventSourceUserData), event.getIntegerValueField(.keyboardEventAutorepeat)))
            if let transition = MediaKeyEvent.decode(event) { media.append(transition) }
        }
    }
    var values: [(CGEventType, Int64, UInt64, Int64, Int64)] { lock.withLock { events } }
    var mediaValues: [MediaKeyEvent.Transition] { lock.withLock { media } }
}

final class AgentInputExecutorTests: XCTestCase {
    func testMediaKeysUseNativeTransitionsAndEmergencyReleases() throws {
        let bounds = CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100))
        for code: UInt16 in [72, 73, 74] {
            let events = CapturedInputEvents()
            let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
            try executor.execute(.keyDown(code: code), bounds: bounds)
            try executor.execute(.keyRepeat(code: code), bounds: bounds)
            XCTAssertEqual(executor.state.keys, [code])
            executor.stop("Emergency")
            XCTAssertEqual(events.mediaValues, [
                .init(code: code, down: true, isRepeat: false),
                .init(code: code, down: true, isRepeat: true),
                .init(code: code, down: false, isRepeat: false)])
            XCTAssertTrue(events.values.allSatisfy { $0.0.rawValue == 14 && $0.3 == AgentInputExecutor.marker })
            XCTAssertTrue(executor.state.keys.isEmpty)
        }
    }

    func testMediaDecoderRejectsOtherSystemEventsAndInvalidTransitions() throws {
        let ordinary = try XCTUnwrap(CGEvent(keyboardEventSource: nil, virtualKey: 72, keyDown: true))
        XCTAssertNil(MediaKeyEvent.decode(ordinary))
        for (subtype, payload) in [(7, 0x0a00), (8, 0x0c00), (8, 0x100a00)] {
            let event = try XCTUnwrap(NSEvent.otherEvent(with: .systemDefined, location: .zero, modifierFlags: [],
                timestamp: 1, windowNumber: 0, context: nil, subtype: Int16(subtype), data1: payload, data2: -1)?.cgEvent)
            XCTAssertNil(MediaKeyEvent.decode(event))
        }
    }

    func testHumanSharingDiscardsStaleDecisionsAndResumesAfterRelease() throws {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        let bounds = CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100))
        let oldRevision = executor.humanInput.revision
        executor.observeHumanInput(InputState(keys: [56]), stopOnInput: false)
        XCTAssertFalse(try executor.execute(.keyDown(code: 0), bounds: bounds, humanRevision: executor.humanInput.revision))
        executor.observeHumanInput(InputState(), stopOnInput: false)
        XCTAssertFalse(try executor.execute(.keyDown(code: 0), bounds: bounds, humanRevision: oldRevision), "Even a completed human press invalidates a pending action.")
        XCTAssertTrue(events.values.isEmpty)
        XCTAssertFalse(executor.isStopped)
        XCTAssertTrue(try executor.execute(.keyDown(code: 0), bounds: bounds, humanRevision: executor.humanInput.revision))
        executor.observeHumanInput(InputState(keys: [56]), stopOnInput: false)
        XCTAssertTrue(executor.isStopped)
        XCTAssertEqual(executor.stopReason, "Human input conflicted with input held by the agent.")
        XCTAssertEqual(events.values.map { $0.0 }, [.keyDown, .keyUp])
        XCTAssertEqual(events.values.map { $0.1 }, [0, 0], "Cleanup releases only agent-owned input, not the human modifier.")
        XCTAssertTrue(executor.state.keys.isEmpty)
    }

    func testHumanStopAlsoHandlesPointerInputWithoutHeldKeys() throws {
        let events = CapturedInputEvents()
        let executor = AgentInputExecutor(capabilities: ActionCapabilities(), emit: events.append)
        executor.observeHumanInput(InputState(cursorX: 12, cursorY: 14), stopOnInput: true)
        XCTAssertTrue(executor.isStopped)
        XCTAssertEqual(executor.stopReason, "Stopped by keyboard or mouse input.")
        XCTAssertTrue(events.values.isEmpty)
    }

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

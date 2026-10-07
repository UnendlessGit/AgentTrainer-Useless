import XCTest
@testable import AgentTrainer

final class CausalAlignmentTests: XCTestCase {
    private func observation(_ time: UInt64) -> VisualObservation {
        VisualObservation(id: time, timeNanoseconds: time, sourceTimeNanoseconds: time, imageFile: "frames/0.jpg", width: 10, height: 10,
                    globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 10, height: 10)), state: InputState(), reusedPixels: false)
    }
    private func event(_ time: UInt64, _ action: ComputerAction = .keyDown(code: 0)) -> InputTransition {
        InputTransition(id: time, timeNanoseconds: time, action: action)
    }

    func testShortPressAndClickSurviveBetweenFrames() throws {
        let samples = try CausalAlignment.align(observations: [observation(10), observation(100)], events: [
            event(11, .keyDown(code: 0)), event(12, .keyUp(code: 0)), event(15, .buttonDown(button: 0)), event(16, .buttonUp(button: 0))
        ])
        XCTAssertEqual(samples[0].eventIndices, [0, 1, 2, 3])
        XCTAssertEqual(samples[1].eventIndices, [])
    }

    func testNeverPairsActionWithSimultaneousOrFutureObservation() throws {
        let samples = try CausalAlignment.align(observations: [observation(10), observation(20), observation(30)],
                                               events: [event(5), event(10), event(19), event(20), event(21), event(31)])
        XCTAssertEqual(samples.map(\.eventIndices), [[2, 3], [4], [5]])
    }

    func testRejectsUnorderedTimelineAndFuturePixels() {
        XCTAssertThrowsError(try CausalAlignment.align(observations: [observation(20), observation(10)], events: []))
        XCTAssertThrowsError(try CausalAlignment.align(observations: [observation(10)], events: [event(30), event(20)]))
        var future = observation(10); future.sourceTimeNanoseconds = 11
        XCTAssertThrowsError(try CausalAlignment.align(observations: [future], events: []))
    }

    func testStaticScenesAndTerminalActionsAreRetained() throws {
        var repeated = observation(20); repeated.sourceTimeNanoseconds = 10; repeated.reusedPixels = true
        let samples = try CausalAlignment.align(observations: [observation(10), repeated], events: [event(100, .keyUp(code: 0))])
        XCTAssertEqual(samples.map(\.eventIndices), [[], [0]])
    }
}

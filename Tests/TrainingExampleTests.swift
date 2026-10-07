import XCTest
@testable import AgentTrainer

final class TrainingExampleTests: XCTestCase {
    func testStreamingTargetsKeepShortTransitionsAndReconstructPriorState() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Sequence", folderID: UUID(), kind: .imitation,
                                                                                  target: CaptureTarget(), settings: RecordingSettings()))
        for time: UInt64 in [10, 100] {
            try journal.append(observation: VisualObservation(id: time, timeNanoseconds: time, sourceTimeNanoseconds: 10,
                imageFile: "frames/0.jpg", width: 10, height: 10, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 10, height: 10)),
                state: InputState(), reusedPixels: time == 100))
        }
        for (time, action): (UInt64, ComputerAction) in [(11, .keyDown(code: 0)), (12, .keyUp(code: 0)), (100, .buttonDown(button: 0)), (101, .buttonUp(button: 0))] {
            try journal.append(event: InputTransition(id: time, timeNanoseconds: time, action: action))
        }
        try journal.finish(at: 150)
        let item = RecordingItem(manifest: journal.snapshot, edits: RecordingEdits(), url: journal.url)
        var examples: [TrainingExample] = []
        try TrainingExampleBuilder.stream(item: item) { examples.append($0) }
        XCTAssertEqual(examples.count, 4)
        XCTAssertEqual(examples.map(\.observation.timeNanoseconds), [10, 10, 10, 100])
        XCTAssertEqual(examples.map(\.state.keys), [[], [0], [], []])
        XCTAssertEqual(examples[3].state.buttons, [0])
        XCTAssertEqual(examples[1].previousAction, .keyDown(code: 0))
        XCTAssertEqual(examples[1].targetDelay, 1e-9, accuracy: 1e-12)
        XCTAssertEqual(examples[2].targetDelay, 88e-9, accuracy: 1e-12)
    }

    func testCoordinateCodecRoundTripAndImpossibleActionMask() throws {
        let capabilities = ActionCapabilities()
        let codec = PolicyActionCodec(capabilities: capabilities)
        let bounds = CaptureRect(CGRect(x: -1920, y: 50, width: 1920, height: 1080))
        let action = ComputerAction.pointer(x: -123, y: 450)
        let coordinates = codec.arguments(for: action, bounds: bounds)
        let token = try XCTUnwrap(codec.token(for: action))
        let decoded = try codec.decode(token: token, x: Double(coordinates.0), y: Double(coordinates.1), delay: 0, bounds: bounds)
        if case .pointer(let x, let y) = decoded {
            XCTAssertEqual(x, -123, accuracy: 0.001); XCTAssertEqual(y, 450, accuracy: 0.001)
        } else { XCTFail("Wrong action family") }
        let release = try XCTUnwrap(codec.token(for: .keyUp(code: 0)))
        XCTAssertLessThan(codec.mask(state: InputState(), capabilities: capabilities)[release], -1e8)
        XCTAssertNil(codec.token(for: .keyDown(code: 127)))
    }
}

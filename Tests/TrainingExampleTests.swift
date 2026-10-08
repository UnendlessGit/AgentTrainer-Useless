import XCTest
@testable import AgentTrainer

final class TrainingExampleTests: XCTestCase {
    func testInvalidSavedTrimsAreIneligibleInsteadOfTrappingDuringConversion() throws {
        var manifest = RecordingManifest(name: "Saved data", folderID: UUID(), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings())
        manifest.status = .complete; manifest.observationCount = 2; manifest.durationNanoseconds = 1_000_000_000
        var item = RecordingItem(manifest: manifest, edits: RecordingEdits(), url: URL(fileURLWithPath: "/unused"))
        for bad in [-1.0, Double.infinity, Double.nan, Double.greatestFiniteMagnitude] {
            item.edits.trimStart = bad
            XCTAssertFalse(item.eligible)
            XCTAssertThrowsError(try TrainingExampleBuilder.stream(item: item) { _ in XCTFail("No rows expected") })
        }
        item.edits.trimStart = 0
        item.edits.trimEnd = 2
        XCTAssertFalse(item.eligible)
        item.edits.trimEnd = 1
        XCTAssertTrue(item.eligible)
        XCTAssertEqual(try item.edits.timeRange(duration: 1).end, 1_000_000_000)
        item.edits.schemaVersion = 99
        XCTAssertFalse(item.eligible)
        XCTAssertThrowsError(try item.edits.timeRange(duration: 1))
        var large = RecordingEdits(); large.trimEnd = 1e30
        XCTAssertThrowsError(try large.timeRange(duration: 1e30))
    }

    func testRecordedRepeatsRemainTimedTargetsWithoutChangingHeldState() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Held Delete", folderID: UUID(), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings()))
        for time: UInt64 in [10, 100] {
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time, sourceTimeNanoseconds: time,
                imageFile: "frames/0.jpg", width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 10, height: 10)),
                state: InputState(), reusedPixels: false))
        }
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 11, action: .keyDown(code: 51)))
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 21, action: .keyDown(code: 51), isRepeat: true))
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 26, action: .keyDown(code: 51), isRepeat: true))
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 30, action: .keyUp(code: 51)))
        try journal.finish(at: 100)
        let item = RecordingItem(manifest: journal.snapshot, edits: RecordingEdits(), url: journal.url)
        var examples: [TrainingExample] = []
        try TrainingExampleBuilder.stream(item: item) { examples.append($0) }
        XCTAssertEqual(examples.map(\.targetAction), [.keyDown(code: 51), .keyRepeat(code: 51), .keyRepeat(code: 51), .keyUp(code: 51)])
        XCTAssertEqual(examples.map(\.state.keys), [[], [51], [51], [51]])
        XCTAssertEqual(examples[2].previousAction, .keyRepeat(code: 51))
        XCTAssertEqual(examples[2].targetDelay, 5e-9, accuracy: 1e-12)
        let capabilities = ActionCapabilities(), codec = PolicyActionCodec(capabilities: capabilities)
        XCTAssertNotNil(codec.token(for: examples[2].targetAction))
        XCTAssertTrue(capabilities.permits(examples[2].targetAction, state: examples[2].state))
        XCTAssertFalse(capabilities.permits(.keyRepeat(code: 51), state: InputState()))
    }

    func testLegacyVocabularyRemainsCompatibleUntilRepeatsAreEnabled() throws {
        var configuration = PolicyConfiguration()
        configuration.capabilities.keyRepeats = nil
        let oldFingerprint = configuration.fingerprint
        let decoded = try JSONDecoder().decode(PolicyConfiguration.self, from: JSONEncoder().encode(configuration))
        XCTAssertFalse(decoded.capabilities.repeatsKeys)
        XCTAssertEqual(decoded.fingerprint, oldFingerprint)
        let original = PolicyActionCodec(capabilities: decoded.capabilities)
        XCTAssertNil(original.token(for: .keyRepeat(code: 51)))
        configuration.capabilities.repeatsKeys = true
        let extended = PolicyActionCodec(capabilities: configuration.capabilities)
        XCTAssertNotEqual(configuration.fingerprint, oldFingerprint)
        XCTAssertEqual(Array(extended.actions.prefix(original.count)), original.actions)
    }

    func testWaitOnlyDemonstrationIsEligibleAndTeachesInaction() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Wait when empty", folderID: UUID(), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings()))
        for time: UInt64 in [10, 100] {
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time, sourceTimeNanoseconds: 10,
                imageFile: "frames/0.jpg", width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 10, height: 10)),
                state: InputState(), reusedPixels: time == 100))
        }
        try journal.finish(at: 150)
        let item = RecordingItem(manifest: journal.snapshot, edits: RecordingEdits(), url: journal.url)
        XCTAssertTrue(item.eligible)
        var actions: [ComputerAction] = []
        try TrainingExampleBuilder.stream(item: item) { actions.append($0.targetAction) }
        XCTAssertEqual(actions.count, 2)
        for action in actions { guard case .wait = action else { return XCTFail("No input should be invented for idle observations") } }
    }

    func testRecoveredTailRequiresReviewAndPreservesOriginalBytes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Interrupted", folderID: UUID(), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings()))
        try Data([0]).write(to: journal.url.appendingPathComponent("frames/0.jpg"))
        for time: UInt64 in [10, 100] {
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time, sourceTimeNanoseconds: time,
                imageFile: "frames/0.jpg", width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 10, height: 10)),
                state: InputState(), reusedPixels: false))
        }
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 11, action: .keyDown(code: 0)))
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 12, action: .keyUp(code: 0)))
        try journal.checkpoint(at: 150, force: true)
        let eventsURL = journal.url.appendingPathComponent("events.jsonl")
        let handle = try FileHandle(forWritingTo: eventsURL)
        try handle.seekToEnd(); try handle.write(contentsOf: Data("{\"torn\":".utf8)); try handle.close()
        let original = try Data(contentsOf: eventsURL)
        var item = RecordingItem(manifest: try RecordingJournal.recover(at: journal.url), edits: RecordingEdits(), url: journal.url)
        XCTAssertFalse(item.eligible)
        XCTAssertThrowsError(try TrainingExampleBuilder.stream(item: item) { _ in })
        item.edits.reviewedRecovery = true
        var examples: [TrainingExample] = []
        try TrainingExampleBuilder.stream(item: item) { examples.append($0) }
        XCTAssertTrue(item.eligible)
        XCTAssertEqual(examples.map(\.targetAction), [.keyDown(code: 0), .keyUp(code: 0)])
        XCTAssertEqual(try Data(contentsOf: eventsURL), original)
        item.manifest.status = .failed
        XCTAssertTrue(item.eligible)
        XCTAssertNoThrow(try TrainingExampleBuilder.stream(item: item) { _ in })
        item.manifest.status = .complete; item.manifest.failure = nil
        XCTAssertThrowsError(try TrainingExampleBuilder.stream(item: item) { _ in })
    }

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

extension TrainingExampleTests {
    func testLongHoldRemainsHeldThroughWaitsUntilRecordedRelease() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try RecordingJournal(root: root, manifest: RecordingManifest(name: "Hold W", folderID: UUID(), kind: .imitation,
            target: CaptureTarget(), settings: RecordingSettings()))
        for tick in 1...15 {
            let time = UInt64(tick) * 100_000_000
            try journal.append(observation: VisualObservation(id: 0, timeNanoseconds: time, sourceTimeNanoseconds: time,
                imageFile: "frames/0.jpg", width: 1, height: 1, globalBounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
                state: InputState(), reusedPixels: false))
        }
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 110_000_000, action: .keyDown(code: 13)))
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 1_310_000_000, action: .keyUp(code: 13)))
        try journal.finish(at: 1_550_000_000)
        let item = RecordingItem(manifest: journal.snapshot, edits: RecordingEdits(), url: journal.url)
        var rows: [TrainingExample] = []
        try TrainingExampleBuilder.stream(item: item) { rows.append($0) }
        XCTAssertEqual(rows.first?.targetAction, .keyDown(code: 13))
        for row in rows where row.decisionTime >= 200_000_000 && row.decisionTime <= 1_300_000_000 {
            XCTAssertEqual(row.state.keys, [13], "Snapshot timing must not erase a recorded hold.")
        }
        XCTAssertTrue(rows.contains { $0.targetAction == .keyUp(code: 13) && $0.state.keys == [13] })
        XCTAssertEqual(rows.last?.state.keys, [])
        XCTAssertEqual(rows.filter { if case .keyDown = $0.targetAction { return true }; return false }.count, 1)
    }
}

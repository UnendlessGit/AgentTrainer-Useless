import XCTest
@testable import AgentTrainer

final class RecordingJournalTests: XCTestCase {
    private var root: URL!
    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try FileManager.default.removeItem(at: root) }

    private func journal() throws -> RecordingJournal {
        try RecordingJournal(root: root, manifest: RecordingManifest(name: "Test", folderID: UUID(), kind: .imitation, target: CaptureTarget(), settings: RecordingSettings()))
    }

    func testInterruptedJournalRecoversCompleteLinesAndDoesNotInventCompletion() throws {
        let journal = try journal()
        try journal.append(event: InputTransition(id: 900, timeNanoseconds: 10, action: .keyDown(code: 0)))
        try journal.append(event: InputTransition(id: 901, timeNanoseconds: 11, action: .keyUp(code: 0)))
        try journal.checkpoint(at: 20, force: true)
        let handle = try FileHandle(forWritingTo: journal.url.appendingPathComponent("events.jsonl"))
        try handle.seekToEnd(); try handle.write(contentsOf: Data("{\"unfinished\":".utf8)); try handle.close()
        let recovered = try RecordingJournal.recover(at: journal.url)
        XCTAssertEqual(recovered.inputEventCount, 2)
        XCTAssertEqual(recovered.status, .interrupted)
        XCTAssertEqual(recovered.durationNanoseconds, 11)
    }

    func testCorruptMiddleEntryIsRejectedRatherThanSilentlySkipped() throws {
        let file = root.appendingPathComponent("events.jsonl")
        try Data("{bad}\n".utf8).write(to: file)
        XCTAssertThrowsError(try JSONLines.read(InputTransition.self, from: file, recoverTail: true) { _ in })
    }

    func testFinishedJournalHasDurableCountsAndRejectsLaterWrites() throws {
        let journal = try journal()
        try journal.append(event: InputTransition(id: 0, timeNanoseconds: 10, action: .keyDown(code: 0)))
        try journal.finish(at: 100)
        let manifest = try AtomicFile.decode(RecordingManifest.self, from: journal.url.appendingPathComponent("manifest.json"))
        XCTAssertEqual(manifest.status, .complete)
        XCTAssertEqual(manifest.durationNanoseconds, 100)
        XCTAssertEqual(manifest.inputEventCount, 1)
        XCTAssertThrowsError(try journal.append(event: InputTransition(id: 1, timeNanoseconds: 101, action: .keyUp(code: 0))))
    }

    func testFailedReplacementPreservesPreviousValidFile() throws {
        struct Failing: Encodable {
            func encode(to encoder: any Encoder) throws { throw DataIntegrityError.io("Injected encoding failure") }
        }
        let url = root.appendingPathComponent("manifest.json")
        try AtomicFile.encode(["version": 1], to: url)
        XCTAssertThrowsError(try AtomicFile.encode(Failing(), to: url))
        XCTAssertEqual(try AtomicFile.decode([String: Int].self, from: url), ["version": 1])
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["manifest.json"])
    }

    func testFramePathRejectsTraversal() {
        XCTAssertTrue(RecordingJournal.isSafeFramePath("frames/00001.jpg"))
        XCTAssertFalse(RecordingJournal.isSafeFramePath("frames/../secret"))
        XCTAssertFalse(RecordingJournal.isSafeFramePath("/tmp/frame.jpg"))
    }
}

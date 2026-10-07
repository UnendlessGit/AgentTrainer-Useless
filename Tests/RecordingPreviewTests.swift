import XCTest
@testable import AgentTrainer

final class RecordingPreviewTests: XCTestCase {
    func testTimelineSeeksBeyondOldPreviewLimitAndBoundsDenseEvents() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = root.appendingPathComponent("events.jsonl")
        XCTAssertTrue(FileManager.default.createFile(atPath: journal.path, contents: nil))
        let writer = try FileHandle(forWritingTo: journal)
        let encoder = JSONEncoder()
        for index in 0..<10_010 {
            let row = InputTransition(id: UInt64(index), timeNanoseconds: UInt64(index * 10), action: .keyDown(code: 0))
            try writer.write(contentsOf: encoder.encode(row)); try writer.write(contentsOf: Data([10]))
        }
        // A recovered journal preserves its torn tail; preview uses complete rows.
        try writer.write(contentsOf: Data("{\"id\":".utf8)); try writer.close()
        let timeline = try RecordingTimeline<InputTransition>(journal: journal, index: root.appendingPathComponent("events.index"))
        XCTAssertEqual(timeline.count, 10_010)
        XCTAssertEqual(try timeline.row(at: 10_005)?.id, 10_005)
        XCTAssertEqual(try timeline.row(at: 0)?.id, 0)
        XCTAssertNil(try timeline.row(at: 10_010))
        let range = try timeline.rows(after: 100_000, through: 100_060, limit: 3)
        XCTAssertEqual(range.total, 6)
        XCTAssertEqual(range.rows.map(\.id), [10_001, 10_002, 10_003])
        XCTAssertEqual(try timeline.rows(after: 100_000, through: 100_000, limit: 100).total, 0)
    }
}

import Foundation
import ImageIO

protocol TimelineRow: Decodable, Sendable {
    var timeNanoseconds: UInt64 { get }
}
extension VisualObservation: TimelineRow {}
extension InputTransition: TimelineRow {}

/// A compact on-disk time/offset index. Neither long recordings nor dense input
/// journals are loaded into memory to seek a frame in the inspector.
struct RecordingTimeline<Row: TimelineRow>: Sendable {
    let journal: URL
    let index: URL
    let count: Int

    init(journal: URL, index: URL) throws {
        self.journal = journal; self.index = index
        let cursor = try JSONLineCursor<Row>(url: journal, recoverTail: true)
        guard FileManager.default.createFile(atPath: index.path, contents: nil) else {
            throw DataIntegrityError.io("Could not create the preview index.")
        }
        let output = try FileHandle(forWritingTo: index)
        defer { try? output.close() }
        var count = 0, lastTime: UInt64 = 0
        while true {
            try Task.checkCancellation()
            let found = try autoreleasepool {
                let position = cursor.offset
                guard let row = try cursor.next() else { return false }
                guard row.timeNanoseconds >= lastTime else { throw DataIntegrityError.invalidTimeline }
                var time = row.timeNanoseconds.littleEndian, offset = position.littleEndian
                try withUnsafeBytes(of: &time) { try output.write(contentsOf: $0) }
                try withUnsafeBytes(of: &offset) { try output.write(contentsOf: $0) }
                lastTime = row.timeNanoseconds; count += 1
                return true
            }
            if !found { break }
        }
        self.count = count
    }

    func row(at index: Int) throws -> Row? {
        guard (0..<count).contains(index) else { return nil }
        let handle = try FileHandle(forReadingFrom: self.index)
        defer { try? handle.close() }
        let entry = try entry(at: index, handle: handle)
        return try JSONLineCursor<Row>(url: journal, recoverTail: true, offset: entry.offset).next()
    }

    func rows(after time: UInt64, through end: UInt64, limit: Int) throws -> (rows: [Row], total: Int) {
        let handle = try FileHandle(forReadingFrom: index)
        defer { try? handle.close() }
        let start = try upperBound(time: time, handle: handle), finish = try upperBound(time: end, handle: handle)
        guard start < finish else { return ([], 0) }
        let position = try entry(at: start, handle: handle).offset
        let cursor = try JSONLineCursor<Row>(url: journal, recoverTail: true, offset: position)
        var rows: [Row] = []
        for _ in 0..<min(max(0, limit), finish - start) {
            try Task.checkCancellation()
            guard let row = try cursor.next() else { throw DataIntegrityError.invalidData("The recording changed while previewing it.") }
            rows.append(row)
        }
        return (rows, finish - start)
    }

    private func upperBound(time: UInt64, handle: FileHandle) throws -> Int {
        var low = 0, high = count
        while low < high {
            let middle = low + (high - low) / 2
            if try entry(at: middle, handle: handle).time <= time { low = middle + 1 } else { high = middle }
        }
        return low
    }

    private func entry(at index: Int, handle: FileHandle) throws -> (time: UInt64, offset: UInt64) {
        try handle.seek(toOffset: UInt64(index) * 16)
        guard let bytes = try handle.read(upToCount: 16), bytes.count == 16 else {
            throw DataIntegrityError.invalidData("The preview index is incomplete. Reopen the inspector.")
        }
        return bytes.withUnsafeBytes {
            (UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)),
             UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self)))
        }
    }
}

struct RecordingPreviewFrame: @unchecked Sendable {
    let observation: VisualObservation
    let image: CGImage
    let events: [InputTransition]
    let eventCount: Int
}

final class RecordingPreview: Sendable {
    let observations: RecordingTimeline<VisualObservation>
    let events: RecordingTimeline<InputTransition>
    private let directory: URL
    private let recording: URL

    init(recording: URL) throws {
        self.recording = recording
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("AgentTrainer-preview-" + UUID().uuidString)
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            observations = try RecordingTimeline(journal: recording.appendingPathComponent("observations.jsonl"), index: directory.appendingPathComponent("observations.index"))
            events = try RecordingTimeline(journal: recording.appendingPathComponent("events.jsonl"), index: directory.appendingPathComponent("events.index"))
        } catch { try? FileManager.default.removeItem(at: directory); throw error }
    }

    deinit {
        let directory = directory
        DispatchQueue.global(qos: .utility).async { try? FileManager.default.removeItem(at: directory) }
    }

    func frame(at index: Int) throws -> RecordingPreviewFrame? {
        try Task.checkCancellation()
        guard let observation = try observations.row(at: index) else { return nil }
        let end = try observations.row(at: index + 1)?.timeNanoseconds ?? UInt64.max
        let inputs = try events.rows(after: observation.timeNanoseconds, through: end, limit: 1000)
        guard RecordingJournal.isSafeFramePath(observation.imageFile),
              let source = CGImageSourceCreateWithURL(recording.appendingPathComponent(observation.imageFile) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary) else {
            throw DataIntegrityError.invalidData("This observation's image could not be decoded.")
        }
        return RecordingPreviewFrame(observation: observation, image: image, events: inputs.rows, eventCount: inputs.total)
    }
}

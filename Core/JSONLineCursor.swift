import Foundation

/// Streaming pull cursor used by dataset merges; memory is bounded to a chunk and
/// one decoded row. Complete recordings require a newline-terminated final entry.
final class JSONLineCursor<Value: Decodable> {
    private let handle: FileHandle
    private let decoder = JSONDecoder()
    private var buffer = Data()
    private var eof = false
    private let recoverTail: Bool
    private(set) var offset: UInt64
    init(url: URL, recoverTail: Bool = false, offset: UInt64 = 0) throws {
        handle = try FileHandle(forReadingFrom: url); self.recoverTail = recoverTail; self.offset = offset
        if offset > 0 { try handle.seek(toOffset: offset) }
    }
    deinit { try? handle.close() }

    func next() throws -> Value? {
        while true {
            if let end = buffer.firstIndex(of: 10) {
                let data = Data(buffer[..<end])
                offset += UInt64(data.count + 1)
                buffer.removeSubrange(...end)
                return try decoder.decode(Value.self, from: data)
            }
            if eof {
                guard buffer.isEmpty || recoverTail else { throw DataIntegrityError.invalidData("The recording journal has an incomplete final entry.") }
                return nil
            }
            guard buffer.count < 1_048_576 else { throw DataIntegrityError.invalidData("A journal entry exceeds the supported size.") }
            if let chunk = try handle.read(upToCount: 65_536), !chunk.isEmpty { buffer.append(chunk) } else { eof = true }
        }
    }
}

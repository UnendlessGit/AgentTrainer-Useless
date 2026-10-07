import Foundation

/// Streaming pull cursor used by dataset merges; memory is bounded to a chunk and
/// one decoded row. Complete recordings require a newline-terminated final entry.
final class JSONLineCursor<Value: Decodable> {
    private let handle: FileHandle
    private let decoder = JSONDecoder()
    private var buffer = Data()
    private var eof = false
    private let recoverTail: Bool
    init(url: URL, recoverTail: Bool = false) throws {
        handle = try FileHandle(forReadingFrom: url); self.recoverTail = recoverTail
    }
    deinit { try? handle.close() }

    func next() throws -> Value? {
        while true {
            if let end = buffer.firstIndex(of: 10) {
                let data = Data(buffer[..<end])
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

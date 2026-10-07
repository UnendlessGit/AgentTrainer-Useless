import Foundation

/// Incremental reader; bounds memory regardless of recording length. Only a torn
/// final line is recoverable. Corruption in a complete line is never skipped.
enum JSONLines {
    static func read<T: Decodable>(_ type: T.Type, from url: URL, recoverTail: Bool = false,
                                    visit: (T) throws -> Void) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let decoder = JSONDecoder()
        var buffer = Data()
        while let chunk = try handle.read(upToCount: 65_536), !chunk.isEmpty {
            buffer.append(chunk)
            while let end = buffer.firstIndex(of: 10) {
                let line = buffer[..<end]
                guard !line.isEmpty else { throw DataIntegrityError.invalidData("An empty journal entry was found.") }
                try visit(decoder.decode(T.self, from: line))
                buffer.removeSubrange(...end)
            }
            guard buffer.count <= 1_048_576 else { throw DataIntegrityError.invalidData("A journal entry exceeds the supported size.") }
        }
        if !buffer.isEmpty && !recoverTail { throw DataIntegrityError.invalidData("The journal ends with an incomplete entry.") }
    }

    static func load<T: Decodable>(_ type: T.Type, from url: URL, limit: Int = 10_000) throws -> [T] {
        var values: [T] = []
        // Continue validation after the preview limit without retaining the rest.
        try read(type, from: url, recoverTail: true) { if values.count < limit { values.append($0) } }
        return values
    }
}

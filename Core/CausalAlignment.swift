import Foundation

enum CausalAlignment {
    struct Sample: Equatable, Sendable {
        var observationIndex: Int
        var eventIndices: [Int]
    }

    /// An event belongs to the latest observation strictly before it. Equal
    /// timestamps are deliberately excluded: callback ordering cannot prove causality.
    /// Terminal events are retained. Empty intervals teach waiting behavior.
    static func align(observations: [VisualObservation], events: [InputTransition]) throws -> [Sample] {
        guard observations.enumerated().allSatisfy({ i, item in
            item.sourceTimeNanoseconds <= item.timeNanoseconds && (i == 0 || observations[i - 1].timeNanoseconds < item.timeNanoseconds)
        }) else { throw DataIntegrityError.invalidTimeline }
        guard events.enumerated().allSatisfy({ i, item in
            i == 0 || events[i - 1].timeNanoseconds <= item.timeNanoseconds
        }) else { throw DataIntegrityError.invalidTimeline }
        var result = observations.indices.map { Sample(observationIndex: $0, eventIndices: []) }
        var cursor = 0
        for (index, event) in events.enumerated() {
            while cursor + 1 < observations.count && observations[cursor + 1].timeNanoseconds < event.timeNanoseconds { cursor += 1 }
            if !observations.isEmpty && observations[cursor].timeNanoseconds < event.timeNanoseconds {
                result[cursor].eventIndices.append(index)
            }
        }
        return result
    }
}

enum DataIntegrityError: LocalizedError {
    case unsupportedVersion(Int), invalidTimeline, invalidData(String), io(String)
    var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let version): "This data uses unsupported version \(version)."
        case .invalidTimeline: "The recording timeline is invalid; training has been stopped."
        case .invalidData(let reason), .io(let reason): reason
        }
    }
}

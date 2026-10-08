import Foundation
import CryptoKit

struct ActionFrequency: Sendable {
    var total = 0
    var inputs = 0
    var actionCounts: [Int: Int] = [:]

    // Distinguish idle, keyboard holds, mouse holds and combined holds. Repeats
    // and releases are much denser during holds than initial presses while idle.
    static func group(for state: InputState) -> Int {
        (state.keys.isEmpty ? 0 : 1) + (state.buttons.isEmpty ? 0 : 2)
    }
}

struct IndexedRecording: Sendable {
    let item: RecordingItem
    let examplesURL: URL
    let offsetsURL: URL
    let count: Int
    var nonWaitCount: Int = 0
    var stateFrequencies = Array(repeating: ActionFrequency(), count: 4)

    func examples(start: Int, count requested: Int) throws -> [TrainingExample] {
        guard start < count else { return [] }
        let offsets = try FileHandle(forReadingFrom: offsetsURL)
        let examples = try FileHandle(forReadingFrom: examplesURL)
        defer { try? offsets.close(); try? examples.close() }
        try offsets.seek(toOffset: UInt64(start * 16))
        let number = min(requested, count - start)
        guard let data = try offsets.read(upToCount: number * 16), data.count == number * 16 else {
            throw DataIntegrityError.invalidData("The training index is incomplete. Prepare the data again.")
        }
        var result: [TrainingExample] = []
        for index in 0..<number {
            let offset = data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: index * 16, as: UInt64.self)) }
            let length = data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: index * 16 + 8, as: UInt64.self)) }
            guard length < 1_048_576 else { throw DataIntegrityError.invalidData("Invalid training index entry.") }
            try examples.seek(toOffset: offset)
            guard let row = try examples.read(upToCount: Int(length)), row.count == Int(length) else {
                throw DataIntegrityError.invalidData("A training example is incomplete.")
            }
            result.append(try JSONDecoder().decode(TrainingExample.self, from: row))
        }
        return result
    }
}

struct SequenceBatchPlan: Sendable {
    var recordings: [IndexedRecording]
    var chunk: Int
    var memoryGroup = 0
    var endsMemory = false
    var sequencePosition = 0
    var warmupChunks = 0
    var resetsMemory: Bool { chunk == 0 }
}

/// Group recordings into persistent lanes. No sequence crosses a recording and
/// recurrent memory carries between adjacent chunks using truncated BPTT.
struct SequenceSchedule {
    let groups: [[IndexedRecording]]
    let groupSteps: [Int]
    let sequenceLength: Int
    let interleaved: Bool
    let groupStarts: [Int]
    // At most eight batch groups retain memory at once, independent of corpus size.
    static let maximumActiveGroups = 8
    var count: Int { groupSteps.reduce(0, +) }

    init(recordings: [IndexedRecording], batchSize: Int, sequenceLength: Int, seed: UInt64, interleaved: Bool = false, rotated: Bool = false) {
        var generator = StableRandom(seed: seed)
        let shuffled = recordings.sorted { $0.item.id.uuidString < $1.item.id.uuidString }.shuffled(using: &generator)
        groups = stride(from: 0, to: shuffled.count, by: batchSize).map { Array(shuffled[$0..<min(shuffled.count, $0 + batchSize)]) }
        groupSteps = groups.map { group in group.map { ($0.count + sequenceLength - 1) / sequenceLength }.max() ?? 0 }
        self.sequenceLength = sequenceLength
        self.interleaved = interleaved
        groupStarts = groupSteps.map { rotated && $0 > 1 ? Int.random(in: 0..<$0, using: &generator) : 0 }
    }

    func plan(at cursor: Int) -> SequenceBatchPlan? {
        guard cursor >= 0, cursor < count else { return nil }
        var remaining = cursor
        if interleaved {
            for start in stride(from: 0, to: groups.count, by: Self.maximumActiveGroups) {
                let end = min(groups.count, start + Self.maximumActiveGroups)
                let steps = Array(groupSteps[start..<end]), total = steps.reduce(0, +)
                if remaining >= total { remaining -= total; continue }
                // Locate the round without materializing a corpus-sized plan array.
                var low = 0, high = steps.max() ?? 0
                while low < high {
                    let middle = (low + high + 1) / 2
                    if steps.reduce(0, { $0 + min($1, middle) }) <= remaining { low = middle }
                    else { high = middle - 1 }
                }
                let chunk = low
                remaining -= steps.reduce(0) { $0 + min($1, chunk) }
                for group in start..<end where chunk < groupSteps[group] {
                    if remaining == 0 {
                        let actualChunk = (groupStarts[group] + chunk) % groupSteps[group]
                        return SequenceBatchPlan(recordings: groups[group], chunk: actualChunk, memoryGroup: group,
                            endsMemory: chunk + 1 == groupSteps[group], sequencePosition: chunk,
                            warmupChunks: chunk == 0 ? groupStarts[group] : 0)
                    }
                    remaining -= 1
                }
                return nil
            }
            return nil
        }
        for (index, steps) in groupSteps.enumerated() {
            if remaining < steps {
                return SequenceBatchPlan(recordings: groups[index], chunk: remaining, memoryGroup: index, endsMemory: remaining + 1 == steps)
            }
            remaining -= steps
        }
        return nil
    }

    /// Groups with partially consumed sequences at this cursor. Used to verify
    /// all carried states on Resume, including a group whose next chunk is later.
    func pendingMemoryGroups(at cursor: Int) -> [Int] {
        guard let next = plan(at: cursor) else { return [] }
        guard interleaved else { return next.resetsMemory ? [] : [next.memoryGroup] }
        let start = next.memoryGroup / Self.maximumActiveGroups * Self.maximumActiveGroups
        return (start..<min(groups.count, start + Self.maximumActiveGroups)).filter { group in
            let consumed = next.sequencePosition + (group < next.memoryGroup ? 1 : 0)
            return consumed > 0 && consumed < groupSteps[group]
        }
    }
}

struct StableRandom: RandomNumberGenerator {
    var seed: UInt64
    mutating func next() -> UInt64 {
        seed &+= 0x9e3779b97f4a7c15
        var z = seed
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}

struct PreparedDataset: Sendable {
    var root: URL
    var training: [IndexedRecording]
    var validation: [IndexedRecording]
    var fingerprint: String
    var excludedOutsideTarget: Int
    var exampleCount: Int { training.reduce(0) { $0 + $1.count } }
    var nonWaitExampleCount: Int { training.reduce(0) { $0 + $1.nonWaitCount } }
    var stateFrequencies: [ActionFrequency] {
        (0..<4).map { group in
            var frequency = ActionFrequency()
            for recording in training {
                let own = recording.stateFrequencies[group]
                frequency.total += own.total; frequency.inputs += own.inputs
                for (token, count) in own.actionCounts { frequency.actionCounts[token, default: 0] += count }
            }
            return frequency
        }
    }

    /// Examples and offsets live on disk; memory use is independent of the number
    /// of frames. A new verified index is built before each run/resume.
    static func prepare(items: [RecordingItem], configuration: PolicyConfiguration, settings: TrainingSettings,
                        stage: TrainingStage, root: URL, checkCancellation: () throws -> Void,
                        progress: (String) -> Void) throws -> PreparedDataset {
        guard !items.isEmpty else { throw DataIntegrityError.invalidData("Assign eligible recordings to this stage in AI Models first.") }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var success = false
        defer { if !success { try? FileManager.default.removeItem(at: root) } }
        var fingerprint = SHA256(), indexed: [IndexedRecording] = [], excluded = 0
        let datasetVersion = stage == .pretraining ? "dataset-v2-wait-futures" : "dataset-v1"
        fingerprint.update(data: Data("\(datasetVersion)|\(stage.rawValue)|\(configuration.fingerprint)|\(settings.seed)|\(settings.validationFraction)".utf8))
        let ignorePointer = stage == .imitation && settings.ignoresPointerMovement
        let ignoreRepeats = stage == .imitation && settings.ignoresKeyRepeats
        if ignorePointer { fingerprint.update(data: Data("|ignore-pointer-v1".utf8)) }
        if stage == .imitation && settings.usesCursorIndependentKeys { fingerprint.update(data: Data("|keyboard-cursor-v1".utf8)) }
        if ignoreRepeats { fingerprint.update(data: Data("|ignore-repeat-v1".utf8)) }
        let codec = PolicyActionCodec(capabilities: configuration.capabilities)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        for item in items.sorted(by: { $0.id.uuidString < $1.id.uuidString }) {
            try checkCancellation()
            if configuration.instructionConditioning {
                do { try ObservationPreprocessor.validateInstruction(item.instruction) }
                catch { throw DataIntegrityError.invalidData("“\(item.name)”: \(error.localizedDescription) Edit its instruction in Library.") }
            }
            progress("Preparing \(item.name)")
            fingerprint.update(data: Data(item.id.uuidString.utf8))
            fingerprint.update(data: try encoder.encode(item.edits))
            // Raw journal bytes are stable even though Codable Set ordering is not.
            for name in ["manifest.json", "observations.jsonl", "events.jsonl"] {
                try hashFile(item.url.appendingPathComponent(name), into: &fingerprint, check: checkCancellation)
            }
            let examplesURL = root.appendingPathComponent(item.id.uuidString + ".jsonl")
            let offsetsURL = root.appendingPathComponent(item.id.uuidString + ".index")
            guard FileManager.default.createFile(atPath: examplesURL.path, contents: nil),
                  FileManager.default.createFile(atPath: offsetsURL.path, contents: nil) else { throw DataIntegrityError.io("Could not create the training index.") }
            let examples = try FileHandle(forWritingTo: examplesURL), offsets = try FileHandle(forWritingTo: offsetsURL)
            defer { try? examples.close(); try? offsets.close() }
            var count = 0, nonWaitCount = 0, position: UInt64 = 0, previousImage = ""
            var frequencies = Array(repeating: ActionFrequency(), count: 4)
            var finalFutureImage: String?
            try TrainingExampleBuilder.stream(item: item, ignoringPointerMovement: ignorePointer, ignoringKeyRepeats: ignoreRepeats) { example in
                try checkCancellation()
                let bounds = example.observation.globalBounds
                guard bounds.isValid else { throw DataIntegrityError.invalidData("\(item.name) has invalid observation geometry.") }
                if example.observation.imageFile != previousImage {
                    try hashFile(item.url.appendingPathComponent(example.observation.imageFile), into: &fingerprint, check: checkCancellation)
                    previousImage = example.observation.imageFile
                }
                switch example.targetAction {
                case .pointer(let x, let y) where !bounds.cgRect.contains(CGPoint(x: x, y: y)):
                    excluded += 1; return
                case .buttonDown, .scroll:
                    if !bounds.cgRect.contains(CGPoint(x: example.state.cursorX, y: example.state.cursorY)) { excluded += 1; return }
                default: break
                }
                guard let token = codec.token(for: example.targetAction),
                      configuration.capabilities.permits(example.targetAction, state: example.state) else {
                    throw DataIntegrityError.invalidData("“\(item.name)” contains \(example.targetAction.label), which this model cannot produce in that input state. Enable the corresponding capability in AI Models, or trim/exclude that part of the recording.")
                }
                if stage == .pretraining {
                    guard example.hasCausalFuture, let next = example.nextObservation else { return }
                    guard RecordingJournal.isSafeFramePath(next.imageFile) else { throw DataIntegrityError.invalidTimeline }
                    finalFutureImage = next.imageFile
                }
                let row = try encoder.encode(example)
                var offset = position.littleEndian, length = UInt64(row.count).littleEndian
                try withUnsafeBytes(of: &offset) { try offsets.write(contentsOf: $0) }
                try withUnsafeBytes(of: &length) { try offsets.write(contentsOf: $0) }
                try examples.write(contentsOf: row); try examples.write(contentsOf: Data([10]))
                position += UInt64(row.count + 1); count += 1
                let group = ActionFrequency.group(for: example.state)
                frequencies[group].total += 1
                if case .wait = example.targetAction {} else {
                    nonWaitCount += 1; frequencies[group].inputs += 1
                    frequencies[group].actionCounts[token, default: 0] += 1
                }
            }
            // A trim may end exactly at the final observation. Its pixels are
            // still a pretraining target even when it has no decision of its own.
            if let finalFutureImage, finalFutureImage != previousImage {
                try hashFile(item.url.appendingPathComponent(finalFutureImage), into: &fingerprint, check: checkCancellation)
            }
            guard count > 0 else { throw DataIntegrityError.invalidData("“\(item.name)” has no usable targets after trimming and capture-boundary checks.") }
            try examples.synchronize(); try offsets.synchronize()
            indexed.append(IndexedRecording(item: item, examplesURL: examplesURL, offsetsURL: offsetsURL, count: count,
                                            nonWaitCount: nonWaitCount, stateFrequencies: frequencies))
        }
        var generator = StableRandom(seed: settings.seed)
        indexed.shuffle(using: &generator)
        let validationCount = indexed.count > 1 && settings.validationFraction > 0
            ? min(indexed.count - 1, max(1, Int((Double(indexed.count) * settings.validationFraction).rounded()))) : 0
        success = true
        return PreparedDataset(root: root, training: Array(indexed.dropFirst(validationCount)), validation: Array(indexed.prefix(validationCount)),
            fingerprint: fingerprint.finalize().map { String(format: "%02x", $0) }.joined(), excludedOutsideTarget: excluded)
    }

    private static func hashFile(_ url: URL, into hash: inout SHA256, check: () throws -> Void) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { try check(); hash.update(data: data) }
    }
}

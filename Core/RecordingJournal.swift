import Foundation

/// Confined to the visual journal worker. Some capture sources deliver complete
/// frames even when their pixels are unchanged. Reuse only byte-identical JPEGs;
/// each observation still retains its own source and availability timestamps.
final class RecordingImageStore {
    private let root: URL
    private var previous: (data: Data, path: String)?

    init(root: URL) { self.root = root }

    func write(_ data: Data, index: UInt64) throws -> String {
        if let previous, previous.data == data { return previous.path }
        let path = String(format: "frames/%012llu.jpg", index)
        try AtomicFile.write(data, to: root.appendingPathComponent(path))
        previous = (data, path)
        return path
    }
}

/// Thread-safe, append-only recording journal. Pixel encoding happens outside this
/// lock; independent input/capture queues only serialize their short append writes.
final class RecordingJournal: @unchecked Sendable {
    let url: URL
    private let lock = NSLock()
    private let events: FileHandle
    private let observations: FileHandle
    private let encoder = JSONEncoder()
    private var manifest: RecordingManifest
    private var closed = false
    private var lastCheckpoint: UInt64 = 0
    private var latestEventTime: UInt64 = 0
    private var latestObservationTime: UInt64 = 0

    init(root: URL, manifest: RecordingManifest) throws {
        self.manifest = manifest
        url = root.appendingPathComponent(manifest.id.uuidString + ".agentrecording", isDirectory: true)
        try FileManager.default.createDirectory(at: url.appendingPathComponent("frames"), withIntermediateDirectories: true)
        let eventURL = url.appendingPathComponent("events.jsonl")
        let observationURL = url.appendingPathComponent("observations.jsonl")
        guard FileManager.default.createFile(atPath: eventURL.path, contents: nil),
              FileManager.default.createFile(atPath: observationURL.path, contents: nil) else {
            throw DataIntegrityError.io("Could not create the recording journal.")
        }
        events = try FileHandle(forWritingTo: eventURL)
        observations = try FileHandle(forWritingTo: observationURL)
        try AtomicFile.encode(manifest, to: url.appendingPathComponent("manifest.json"))
    }

    deinit { try? events.close(); try? observations.close() }

    var snapshot: RecordingManifest { lock.withLock { manifest } }

    func setInitialInputState(_ state: InputState) throws {
        try lock.withLock {
            guard !closed, manifest.observationCount == 0 else { throw DataIntegrityError.invalidData("Initial input state must be saved before observations.") }
            manifest.initialInputState = state
            try AtomicFile.encode(manifest, to: url.appendingPathComponent("manifest.json"))
        }
    }

    func append(event: InputTransition) throws {
        try lock.withLock {
            guard !closed else { throw DataIntegrityError.io("Recording is already closed.") }
            guard manifest.inputEventCount == 0 || event.timeNanoseconds >= latestEventTime else {
                throw DataIntegrityError.invalidTimeline
            }
            var value = event
            value.id = UInt64(manifest.inputEventCount)
            try append(value, to: events)
            latestEventTime = value.timeNanoseconds
            manifest.inputEventCount += 1
            manifest.durationNanoseconds = max(manifest.durationNanoseconds, value.timeNanoseconds)
        }
    }

    func append(observation: VisualObservation) throws {
        try lock.withLock {
            guard !closed else { throw DataIntegrityError.io("Recording is already closed.") }
            guard observation.sourceTimeNanoseconds <= observation.timeNanoseconds,
                  manifest.observationCount == 0 || observation.timeNanoseconds > latestObservationTime else {
                throw DataIntegrityError.invalidTimeline
            }
            var value = observation
            value.id = UInt64(manifest.observationCount)
            try append(value, to: observations)
            latestObservationTime = value.timeNanoseconds
            manifest.observationCount += 1
            manifest.durationNanoseconds = max(manifest.durationNanoseconds, value.timeNanoseconds)
        }
    }

    func noteDroppedVisualFrame() { lock.withLock { manifest.droppedVisualFrames += 1 } }

    func checkpoint(at time: UInt64, force: Bool = false) throws {
        try lock.withLock {
            guard !closed else { return }
            guard force || time >= lastCheckpoint + 1_000_000_000 else { return }
            try events.synchronize()
            try observations.synchronize()
            manifest.durationNanoseconds = max(manifest.durationNanoseconds, time)
            try AtomicFile.encode(manifest, to: url.appendingPathComponent("manifest.json"))
            lastCheckpoint = time
        }
    }

    func finish(at time: UInt64, failure: String? = nil) throws {
        try lock.withLock {
            guard !closed else { return }
            try events.synchronize()
            try observations.synchronize()
            manifest.status = failure == nil ? .complete : .failed
            manifest.failure = failure
            manifest.durationNanoseconds = max(manifest.durationNanoseconds, time)
            try AtomicFile.encode(manifest, to: url.appendingPathComponent("manifest.json"))
            closed = true
            try events.close()
            try observations.close()
        }
    }

    private func append<T: Encodable>(_ value: T, to file: FileHandle) throws {
        var data = try encoder.encode(value)
        data.append(10)
        try file.write(contentsOf: data)
    }

    static func recover(at url: URL) throws -> RecordingManifest {
        var manifest = try AtomicFile.decode(RecordingManifest.self, from: url.appendingPathComponent("manifest.json"))
        guard manifest.schemaVersion == 1 else { throw DataIntegrityError.unsupportedVersion(manifest.schemaVersion) }
        guard manifest.status == .recording else { return manifest }
        var observationCount = 0, eventCount = 0
        var lastObservation: UInt64 = 0, lastEvent: UInt64 = 0
        try JSONLines.read(VisualObservation.self, from: url.appendingPathComponent("observations.jsonl"), recoverTail: true) { item in
            guard item.id == UInt64(observationCount), item.sourceTimeNanoseconds <= item.timeNanoseconds,
                  observationCount == 0 || item.timeNanoseconds > lastObservation,
                  isSafeFramePath(item.imageFile),
                  FileManager.default.fileExists(atPath: url.appendingPathComponent(item.imageFile).path) else {
                throw DataIntegrityError.invalidData("Recovery found missing pixels or an invalid observation.")
            }
            observationCount += 1
            lastObservation = item.timeNanoseconds
        }
        try JSONLines.read(InputTransition.self, from: url.appendingPathComponent("events.jsonl"), recoverTail: true) { item in
            guard item.id == UInt64(eventCount), eventCount == 0 || item.timeNanoseconds >= lastEvent else {
                throw DataIntegrityError.invalidTimeline
            }
            eventCount += 1
            lastEvent = item.timeNanoseconds
        }
        manifest.observationCount = observationCount
        manifest.inputEventCount = eventCount
        manifest.durationNanoseconds = max(lastObservation, lastEvent)
        manifest.status = .interrupted
        manifest.failure = "Recovered after interruption. Review before using this recording."
        try AtomicFile.encode(manifest, to: url.appendingPathComponent("manifest.json"))
        return manifest
    }

    static func isSafeFramePath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts[0] == "frames" && !parts[1].isEmpty && !parts[1].contains("..")
    }
}

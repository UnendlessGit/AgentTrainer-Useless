import Foundation

struct TrainingExample: Codable, Sendable {
    var recordingID: UUID
    var observation: VisualObservation
    var state: InputState
    var previousAction: ComputerAction
    var decisionTime: UInt64
    var elapsedSincePreviousAction: Double
    var targetAction: ComputerAction
    var targetDelay: Double
    var nextObservation: VisualObservation?
    var instruction: String
}

enum TrainingExampleBuilder {
    /// Merge both journals in timestamp order. Input state is reconstructed from
    /// raw transitions (not from possibly delayed callback snapshots). Every
    /// sub-frame press/release remains a separate autoregressive training target.
    static func stream(item: RecordingItem, visit: (TrainingExample) throws -> Void) throws {
        guard item.eligible else { throw DataIntegrityError.invalidData("“\(item.name)” is not eligible for training.") }
        guard item.manifest.schemaVersion == 1, item.manifest.actionSchemaVersion == ComputerAction.schemaVersion else {
            throw DataIntegrityError.invalidData("The recording's observation/action version is incompatible with this training pipeline.")
        }
        let recovered = item.needsRecoveryReview && item.edits.reviewedRecovery == true
        let observations = try JSONLineCursor<VisualObservation>(url: item.url.appendingPathComponent("observations.jsonl"), recoverTail: recovered)
        let events = try JSONLineCursor<InputTransition>(url: item.url.appendingPathComponent("events.jsonl"), recoverTail: recovered)
        var event = try events.next()
        var current = try observations.next()
        var state = item.manifest.initialInputState
        var previousAction: ComputerAction = .wait(seconds: 0)
        var previousActionTime: UInt64 = 0
        var lastEventTime: UInt64 = 0
        let trimStart = UInt64(item.edits.trimStart * 1e9)
        let trimEnd = UInt64((item.edits.trimEnd ?? item.manifest.duration) * 1e9)

        func consume(_ input: InputTransition) throws {
            guard input.timeNanoseconds >= lastEventTime else { throw DataIntegrityError.invalidTimeline }
            lastEventTime = input.timeNanoseconds
            state.apply(input.action)
            if let x = input.cursorX, let y = input.cursorY { state.cursorX = x; state.cursorY = y }
            previousAction = input.learningAction
            previousActionTime = input.timeNanoseconds
            event = try events.next()
        }

        while let observation = current {
            guard observation.sourceTimeNanoseconds <= observation.timeNanoseconds,
                  RecordingJournal.isSafeFramePath(observation.imageFile) else { throw DataIntegrityError.invalidTimeline }
            let next = try observations.next()
            if let next, next.timeNanoseconds <= observation.timeNanoseconds { throw DataIntegrityError.invalidTimeline }
            let intervalEnd = min(next?.timeNanoseconds ?? item.manifest.durationNanoseconds, trimEnd)
            // Events with no causally earlier observation are context only.
            while let input = event, input.timeNanoseconds <= observation.timeNanoseconds { try consume(input) }
            if !item.manifest.settings.pointerMovement && !item.manifest.settings.relativeMovement {
                state.cursorX = observation.state.cursorX; state.cursorY = observation.state.cursorY
            }
            let future = next.flatMap { $0.timeNanoseconds <= trimEnd ? $0 : nil }
            var decisionTime = max(observation.timeNanoseconds, previousActionTime)
            var emitted = false
            while let input = event, input.timeNanoseconds <= intervalEnd {
                if input.timeNanoseconds >= trimStart && observation.timeNanoseconds >= trimStart {
                    try visit(TrainingExample(recordingID: item.id, observation: observation, state: state,
                        previousAction: previousAction, decisionTime: decisionTime,
                        elapsedSincePreviousAction: Double(decisionTime - min(decisionTime, previousActionTime)) / 1e9,
                        targetAction: input.learningAction, targetDelay: Double(input.timeNanoseconds - decisionTime) / 1e9,
                        nextObservation: future, instruction: item.instruction))
                    emitted = true
                }
                try consume(input)
                decisionTime = max(decisionTime, input.timeNanoseconds)
            }
            if !emitted && observation.timeNanoseconds >= trimStart && intervalEnd > decisionTime {
                let delay = Double(intervalEnd - decisionTime) / 1e9
                try visit(TrainingExample(recordingID: item.id, observation: observation, state: state,
                    previousAction: previousAction, decisionTime: decisionTime,
                    elapsedSincePreviousAction: Double(decisionTime - min(decisionTime, previousActionTime)) / 1e9,
                    targetAction: .wait(seconds: delay), targetDelay: delay, nextObservation: future, instruction: item.instruction))
                // Waiting is an executed action too; train the same preceding-action
                // semantics that the live executor supplies at the next decision.
                previousAction = .wait(seconds: delay); previousActionTime = intervalEnd
            }
            if intervalEnd == trimEnd { break }
            current = next
        }
    }
}

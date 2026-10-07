import Foundation
import CoreGraphics
import MLX
import MLXNN

enum BatchField: Int, CaseIterable {
    case images, crops, context, previousActions, instructions, actions, arguments, delay, spatial, offsets
    case pointerMask, continuousMask, valid, actionMask, futurePixels, currentPixels, futureMask
}

struct TrainingBatch {
    var arrays: [MLXArray]
    var validCount: Int

    static func load(plan: SequenceBatchPlan, configuration c: PolicyConfiguration, stage: TrainingStage,
                     checkCancellation: () throws -> Void) throws -> TrainingBatch {
        let codec = PolicyActionCodec(capabilities: c.capabilities), length = c.sequenceLength
        var images: [MLXArray] = [], crops: [MLXArray] = [], future: [MLXArray] = []
        var contexts: [Float] = [], previous: [Int32] = [], instructions: [Int32] = [], actions: [Int32] = []
        var arguments: [Float] = [], delays: [Int32] = [], spatial: [Int32] = [], offsets: [Float] = []
        var pointer: [Float] = [], continuous: [Float] = [], valid: [Float] = [], masks: [Float] = [], futureMask: [Float] = []
        let zero = MLXArray.zeros([c.imageSize, c.imageSize, 3]), grid = c.imageSize / c.patchSize
        var count = 0
        for recording in plan.recordings {
            let rows = try recording.examples(start: plan.chunk * length, count: length)
            // Two image slots cover consecutive static observations without an
            // unbounded decoded-frame or tensor cache.
            var cache: [(String, CGImage, MLXArray)] = []
            func read(_ name: String) throws -> (CGImage, MLXArray) {
                guard RecordingJournal.isSafeFramePath(name) else { throw DataIntegrityError.invalidData("Invalid observation image path.") }
                if let entry = cache.first(where: { $0.0 == name }) { return (entry.1, entry.2) }
                let image = try ObservationPreprocessor.image(at: recording.item.url.appendingPathComponent(name))
                let pixels = try ObservationPreprocessor.pixels(image, size: c.imageSize)
                cache.append((name, image, pixels)); if cache.count > 2 { cache.removeFirst() }
                return (image, pixels)
            }
            for time in 0..<length {
                try checkCancellation()
                guard time < rows.count else {
                    images.append(zero); crops.append(zero); future.append(zero)
                    contexts += Array(repeating: 0, count: PolicyNetwork.contextSize)
                    previous.append(0); instructions += Array(repeating: 0, count: PolicyNetwork.instructionLength)
                    actions.append(0); arguments += [0, 0]; delays.append(0); spatial.append(0); offsets += [0, 0]
                    pointer.append(0); continuous.append(0); valid.append(0); masks += Array(repeating: 0, count: codec.count); futureMask.append(0)
                    continue
                }
                let row = rows[time], observation = row.observation
                let (image, pixels) = try read(observation.imageFile)
                images.append(pixels)
                crops.append(c.detailCrop ? try ObservationPreprocessor.detailCrop(image, state: row.state, bounds: observation.globalBounds, size: c.imageSize) : zero)
                contexts += ObservationPreprocessor.context(state: row.state, bounds: observation.globalBounds,
                    previousAction: row.previousAction, elapsed: row.elapsedSincePreviousAction,
                    sourceAge: Double(row.decisionTime - min(row.decisionTime, observation.sourceTimeNanoseconds)) / 1e9)
                previous.append(Int32(codec.token(for: row.previousAction) ?? codec.count))
                instructions += ObservationPreprocessor.instruction(row.instruction)
                guard let token = codec.token(for: row.targetAction) else { throw DataIntegrityError.invalidData("An indexed action is incompatible with this model.") }
                actions.append(Int32(token)); masks += codec.mask(state: row.state, capabilities: c.capabilities)
                let (x, y) = codec.arguments(for: row.targetAction, bounds: observation.globalBounds)
                arguments += [x, y]
                delays.append(Int32(PolicyNetwork.delayBins.indices.min(by: { abs(PolicyNetwork.delayBins[$0] - row.targetDelay) < abs(PolicyNetwork.delayBins[$1] - row.targetDelay) }) ?? 0))
                let sx = min(Float(grid) - 0.00001, max(0, x * Float(grid)))
                let sy = min(Float(grid) - 0.00001, max(0, y * Float(grid)))
                spatial.append(Int32(Int(sy) * grid + Int(sx))); offsets += [sx - floor(sx), sy - floor(sy)]
                if case .pointer = row.targetAction { pointer.append(1) } else { pointer.append(0) }
                switch row.targetAction { case .relativePointer, .scroll: continuous.append(1); default: continuous.append(0) }
                valid.append(1); count += 1
                if stage == .pretraining, let next = row.nextObservation {
                    future.append(try read(next.imageFile).1)
                    // A non-static capture whose pixels precede the action is not
                    // evidence of that action's outcome.
                    let actionTime = row.decisionTime + UInt64(max(0, row.targetDelay) * 1e9)
                    futureMask.append(next.reusedPixels || next.sourceTimeNanoseconds >= actionTime ? 1 : 0)
                } else { future.append(zero); futureMask.append(0) }
            }
        }
        let b = plan.recordings.count, t = length, imageShape = [b, t, c.imageSize, c.imageSize, 3]
        let imageArray = stacked(images).reshaped(imageShape)
        func patchPixels(_ pixels: MLXArray) -> MLXArray {
            mean(pixels.reshaped([b, t, grid, c.patchSize, grid, c.patchSize, 3]), axes: [3, 5]).reshaped([b, t, grid * grid, 3])
        }
        let arrays = [imageArray, stacked(crops).reshaped(imageShape), MLXArray(contexts, [b, t, PolicyNetwork.contextSize]),
            MLXArray(previous, [b, t]), MLXArray(instructions, [b, t, PolicyNetwork.instructionLength]), MLXArray(actions, [b, t]),
            MLXArray(arguments, [b, t, 2]), MLXArray(delays, [b, t]), MLXArray(spatial, [b, t]), MLXArray(offsets, [b, t, 2]),
            MLXArray(pointer, [b, t]), MLXArray(continuous, [b, t]), MLXArray(valid, [b, t]), MLXArray(masks, [b, t, codec.count]),
            stage == .pretraining ? patchPixels(stacked(future).reshaped(imageShape)) : MLXArray.zeros([b, t, grid * grid, 3]),
            stage == .pretraining ? patchPixels(imageArray) : MLXArray.zeros([b, t, grid * grid, 3]), MLXArray(futureMask, [b, t])]
        return TrainingBatch(arrays: arrays, validCount: count)
    }
}

enum PolicyLoss {
    static func forward(_ model: PolicyNetwork, _ arrays: [MLXArray]) -> PolicyForward {
        func a(_ field: BatchField) -> MLXArray { arrays[field.rawValue] }
        return model(images: a(.images), crops: model.configuration.detailCrop ? a(.crops) : nil, context: a(.context),
            previousActions: a(.previousActions), instructions: a(.instructions), dynamicsActions: a(.actions),
            dynamicsArguments: a(.arguments), hidden: Array(arrays.dropFirst(BatchField.allCases.count)))
    }

    static func values(_ model: PolicyNetwork, _ arrays: [MLXArray], stage: TrainingStage) -> [MLXArray] {
        let output = forward(model, arrays)
        return [loss(output, arrays, stage: stage)] + output.hidden.map { stopGradient($0) }
    }

    static func loss(_ output: PolicyForward, _ arrays: [MLXArray], stage: TrainingStage) -> MLXArray {
        func a(_ field: BatchField) -> MLXArray { arrays[field.rawValue] }
        let loss: MLXArray
        if stage == .pretraining {
            let valid = a(.valid) * a(.futureMask)
            // Emphasize changed patches while retaining an absolute RGB target.
            // The condition is the recorded action; no imitation-head loss occurs.
            let change = mean(abs(a(.futurePixels) - a(.currentPixels)), axis: -1)
            let weights = 1 + 4 * minimum(change / 0.1, 1)
            let perStep = mean(mean(square(output.futurePixels - a(.futurePixels)), axis: -1) * weights, axis: -1)
            loss = sum(perStep * valid) / maximum(sum(valid), 1)
        } else {
            let valid = a(.valid)
            let action = crossEntropy(logits: output.actionLogits + a(.actionMask), targets: a(.actions))
            let delayLogits = takeAlong(output.delayLogits, a(.actions).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let timing = crossEntropy(logits: delayLogits, targets: a(.delay))
            let spatial = crossEntropy(logits: output.spatialLogits, targets: a(.spatial))
            let selectedOffset = takeAlong(output.spatialOffsets, a(.spatial).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let offset = mean(square(selectedOffset - a(.offsets)), axis: -1)
            let selectedArguments = takeAlong(output.continuousArguments, a(.actions).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let arguments = mean(square(selectedArguments - a(.arguments)), axis: -1)
            let perStep = action + 0.25 * timing + a(.pointerMask) * (spatial + 2 * offset) + a(.continuousMask) * arguments
            loss = sum(perStep * valid) / maximum(sum(valid), 1)
        }
        return loss
    }
}

/// Counts aggregate across recordings without averaging unequal-sized batches.
/// Non-wait accuracy exposes a policy that achieves high accuracy by doing nothing.
struct ActionEvaluation: Codable, Equatable, Sendable {
    struct ActionCounts: Codable, Equatable, Identifiable, Sendable {
        var token: Int
        var action: ComputerAction
        var correct: Int
        var targets: Int
        var predictions: Int
        var id: Int { token }
    }

    var correct = 0
    var total = 0
    var nonWaitCorrect = 0
    var nonWaitTotal = 0
    var nonWaitPredictions = 0
    // Optional so older checkpoint manifests remain readable.
    var actionBreakdown: [ActionCounts]?
    var accuracy: Double? { total > 0 ? Double(correct) / Double(total) : nil }
    var nonWaitAccuracy: Double? { nonWaitTotal > 0 ? Double(nonWaitCorrect) / Double(nonWaitTotal) : nil }
    var nonWaitPrecision: Double? { nonWaitPredictions > 0 ? Double(nonWaitCorrect) / Double(nonWaitPredictions) : nil }

    mutating func add(_ other: Self) {
        if total == 0 {
            actionBreakdown = other.actionBreakdown
        } else if other.total > 0 {
            if let own = actionBreakdown, let incoming = other.actionBreakdown {
                var merged = Dictionary(uniqueKeysWithValues: own.map { ($0.token, $0) })
                for row in incoming {
                    if var old = merged[row.token] {
                        old.correct += row.correct; old.targets += row.targets; old.predictions += row.predictions
                        merged[row.token] = old
                    } else { merged[row.token] = row }
                }
                actionBreakdown = merged.values.sorted { $0.token < $1.token }
            } else { actionBreakdown = nil } // Never display a partial breakdown as the whole evaluation.
        }
        correct += other.correct; total += other.total; nonWaitCorrect += other.nonWaitCorrect
        nonWaitTotal += other.nonWaitTotal; nonWaitPredictions += other.nonWaitPredictions
    }

    static func measure(logits: MLXArray, targets: MLXArray, valid: MLXArray, mask: MLXArray,
                        actions: [ComputerAction]) -> Self {
        let prediction = argMax(logits + mask, axis: -1)
        let tokens = MLXArray((0..<actions.count).map(Int32.init))
        let targetColumns = (targets.reshaped([-1, 1]) .== tokens).asType(.float32)
        let predictionColumns = (prediction.reshaped([-1, 1]) .== tokens).asType(.float32)
        let weights = valid.reshaped([-1, 1])
        // Reduce on Metal and read only three counts per vocabulary entry.
        let counts = stacked([sum(targetColumns * predictionColumns * weights, axis: 0),
                              sum(targetColumns * weights, axis: 0), sum(predictionColumns * weights, axis: 0)])
            .asArray(Float.self)
        let n = actions.count
        let rows = actions.enumerated().map { token, action in
            ActionCounts(token: token, action: action, correct: Int(counts[token]),
                         targets: Int(counts[n + token]), predictions: Int(counts[2 * n + token]))
        }
        let nonWait = rows.dropFirst() // The shared codec assigns token zero to wait.
        return Self(correct: rows.reduce(0) { $0 + $1.correct }, total: rows.reduce(0) { $0 + $1.targets },
                    nonWaitCorrect: nonWait.reduce(0) { $0 + $1.correct }, nonWaitTotal: nonWait.reduce(0) { $0 + $1.targets },
                    nonWaitPredictions: nonWait.reduce(0) { $0 + $1.predictions },
                    actionBreakdown: rows.filter { $0.targets > 0 || $0.predictions > 0 })
    }
}

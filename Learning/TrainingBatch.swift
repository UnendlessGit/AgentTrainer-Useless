import Foundation
import CoreGraphics
import MLX
import MLXNN

enum BatchField: Int, CaseIterable {
    case images, crops, context, previousActions, instructions, actions, arguments, delay, spatial, offsets
    case pointerMask, continuousMask, valid, actionMask, futurePixels, currentPixels, futureMask, actionWeights
}

struct TrainingBatch {
    var arrays: [MLXArray]
    var validCount: Int
    var gradientWeightScale: Float = 1

    static func load(plan: SequenceBatchPlan, configuration c: PolicyConfiguration, stage: TrainingStage,
                     actionBalance: PolicyLoss.ActionBalance? = nil, capabilities: ActionCapabilities? = nil,
                     cursorIndependent: Bool = false,
                     checkCancellation: () throws -> Void) throws -> TrainingBatch {
        let codec = PolicyActionCodec(capabilities: c.capabilities), length = c.sequenceLength
        var images: [MLXArray] = [], crops: [MLXArray] = [], future: [MLXArray] = []
        var contexts: [Float] = [], previous: [Int32] = [], instructions: [Int32] = [], actions: [Int32] = []
        var arguments: [Float] = [], delays: [Int32] = [], spatial: [Int32] = [], offsets: [Float] = []
        var pointer: [Float] = [], continuous: [Float] = [], valid: [Float] = [], masks: [Float] = [], futureMask: [Float] = []
        var actionWeights: [Float] = []
        let zero = MLXArray.zeros([c.imageSize, c.imageSize, 3]), grid = c.imageSize / c.patchSize
        var count = 0, importance: Float = 0
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
                    actionWeights += [0, 0]
                    continue
                }
                let row = rows[time], observation = row.observation
                let (image, pixels) = try read(observation.imageFile)
                images.append(pixels)
                crops.append(c.detailCrop && !cursorIndependent ? try ObservationPreprocessor.detailCrop(image, state: row.state, bounds: observation.globalBounds, size: c.imageSize) : zero)
                let modelState = ObservationPreprocessor.modelState(row.state, bounds: observation.globalBounds, cursorIndependent: cursorIndependent)
                contexts += ObservationPreprocessor.context(state: modelState, bounds: observation.globalBounds,
                    previousAction: row.previousAction, elapsed: row.elapsedSincePreviousAction,
                    sourceAge: Double(row.decisionTime - min(row.decisionTime, observation.sourceTimeNanoseconds)) / 1e9)
                previous.append(Int32(codec.token(for: row.previousAction) ?? codec.count))
                instructions += try ObservationPreprocessor.instruction(c.instructionConditioning ? row.instruction : "")
                guard let token = codec.token(for: row.targetAction) else { throw DataIntegrityError.invalidData("An indexed action is incompatible with this model.") }
                actions.append(Int32(token)); masks += codec.mask(state: row.state, capabilities: capabilities ?? c.capabilities)
                let balance = actionBalance?.forState(row.state, recordingID: recording.item.id)
                let gateWeight = token == 0 ? (balance?.waitWeight ?? 1) : (balance?.inputWeight ?? 1)
                let choiceWeight = balance?.weight(for: token) ?? 1
                actionWeights += [gateWeight, choiceWeight]
                importance += gateWeight + (token == 0 ? 0 : choiceWeight)
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
                    futureMask.append(row.hasCausalFuture ? 1 : 0)
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
            stage == .pretraining ? patchPixels(imageArray) : MLXArray.zeros([b, t, grid * grid, 3]), MLXArray(futureMask, [b, t]),
            MLXArray(actionWeights, [b, t, 2])]
        // The balanced gate and conditional choice each carry one unit of
        // expected sample importance. Normalize before clipping, then retain
        // that importance in Adam's moments. A fixed bound otherwise erases
        // rare-event weights while leaving easy wait-only gradients intact.
        let scale = stage == .imitation && actionBalance?.balancesTokens == true && count > 0
            ? max(1e-3, importance / Float(count) / 2) : 1
        return TrainingBatch(arrays: arrays, validCount: count, gradientWeightScale: scale)
    }
}

enum PolicyLoss {
    struct ActionBalance {
        let waitWeight: Float
        let inputWeight: Float
        let choiceWeight: Float
        var states: [ActionBalance] = []
        var tokenWeights: [Int: Float] = [:]
        var recordingStates: [UUID: [ActionBalance]] = [:]
        var balancesTokens = false

        init(total: Int, inputs: Int) {
            let waits = total - inputs
            let groups: Float = waits > 0 && inputs > 0 ? 2 : 1
            // An unseen class may occur in validation. It must still contribute
            // loss instead of receiving a misleading zero score.
            waitWeight = waits > 0 ? Float(total) / (groups * Float(waits)) : 1
            inputWeight = inputs > 0 ? Float(total) / (groups * Float(inputs)) : 1
            choiceWeight = inputs > 0 ? Float(total) / Float(inputs) : 1
        }

        init(frequencies: [ActionFrequency], balanceTokens: Bool = false, recordings: [IndexedRecording] = []) {
            self.init(total: frequencies.reduce(0) { $0 + $1.total }, inputs: frequencies.reduce(0) { $0 + $1.inputs })
            balancesTokens = balanceTokens
            let fallback = self
            states = frequencies.map { frequency in
                var balance = frequency.total > 0 ? ActionBalance(total: frequency.total, inputs: frequency.inputs) : fallback
                if balanceTokens {
                    let observed = frequency.actionCounts.filter { $0.value > 0 }
                    for (token, count) in observed {
                        balance.tokenWeights[token] = Float(frequency.total) / Float(observed.count * count)
                    }
                }
                return balance
            }
            for recording in recordings {
                let own = ActionBalance(frequencies: recording.stateFrequencies, balanceTokens: balanceTokens)
                recordingStates[recording.item.id] = own.states
            }
        }

        func forState(_ state: InputState, recordingID: UUID? = nil) -> ActionBalance {
            let group = ActionFrequency.group(for: state)
            if let recordingID, let own = recordingStates[recordingID] { return own[group] }
            return states.isEmpty ? self : states[group]
        }

        func weight(for token: Int) -> Float { tokenWeights[token] ?? choiceWeight }
    }

    static func forward(_ model: PolicyNetwork, _ arrays: [MLXArray], wholeSceneDetail: Bool = false) -> PolicyForward {
        func a(_ field: BatchField) -> MLXArray { arrays[field.rawValue] }
        return model(images: a(.images), crops: model.configuration.detailCrop ? a(.crops) : nil, context: a(.context),
            previousActions: a(.previousActions), instructions: a(.instructions), dynamicsActions: a(.actions),
            dynamicsArguments: a(.arguments), hidden: Array(arrays.dropFirst(BatchField.allCases.count)), wholeSceneDetail: wholeSceneDetail)
    }

    static func values(_ model: PolicyNetwork, _ arrays: [MLXArray], stage: TrainingStage, balanceInputChoices: Bool = false,
                       actionBalance: ActionBalance? = nil, wholeSceneDetail: Bool = false) -> [MLXArray] {
        let output = forward(model, arrays, wholeSceneDetail: wholeSceneDetail)
        return [loss(output, arrays, stage: stage, balanceInputChoices: balanceInputChoices, actionBalance: actionBalance)] + output.hidden.map { stopGradient($0) }
    }

    /// Balanced runs weight both the wait/input gate and conditional input choice
    /// using training-set frequencies. The legacy choice-only option preserves
    /// the recorded gate frequency for exact checkpoint resume.
    static func actionLoss(logits: MLXArray, targets: MLXArray, valid: MLXArray,
                           balanceInputChoices: Bool, actionBalance: ActionBalance? = nil, sampleWeights: MLXArray? = nil) -> MLXArray {
        if let balance = actionBalance, logits.dim(-1) > 1 {
            let input = logits[.ellipsis, 1...]
            let isInput = (targets .> 0).asType(.int32)
            let gate = stacked([logits[.ellipsis, 0], logSumExp(input, axis: -1)], axis: -1)
            let inputMask = isInput.asType(.float32)
            let weights = sampleWeights?[.ellipsis, 0] ?? ((1 - inputMask) * balance.waitWeight + inputMask * balance.inputWeight)
            let gateLoss = crossEntropy(logits: gate, targets: isInput) * weights
            let choice = crossEntropy(logits: input, targets: maximum(targets - 1, 0))
            // Dataset-wide weights keep rare transitions influential even in
            // long runs of wait-only chunks. Reductions compose across batches.
            let choiceWeight = sampleWeights?[.ellipsis, 1] ?? MLXArray(balance.choiceWeight)
            return sum((gateLoss + choice * inputMask * choiceWeight) * valid) / maximum(sum(valid), 1)
        }
        guard balanceInputChoices, logits.dim(-1) > 1 else {
            return sum(crossEntropy(logits: logits, targets: targets) * valid) / maximum(sum(valid), 1)
        }
        let input = logits[0..., 0..., 1...]
        let isInput = (targets .> 0).asType(.int32)
        let gate = stacked([logits[0..., 0..., 0], logSumExp(input, axis: -1)], axis: -1)
        let gateLoss = sum(crossEntropy(logits: gate, targets: isInput) * valid) / maximum(sum(valid), 1)
        let inputValid = isInput.asType(.float32) * valid
        let choice = crossEntropy(logits: input, targets: maximum(targets - 1, 0))
        return gateLoss + sum(choice * inputValid) / maximum(sum(inputValid), 1)
    }

    static func loss(_ output: PolicyForward, _ arrays: [MLXArray], stage: TrainingStage, balanceInputChoices: Bool = false,
                     actionBalance: ActionBalance? = nil) -> MLXArray {
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
            let delayLogits = takeAlong(output.delayLogits, a(.actions).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let timing = crossEntropy(logits: delayLogits, targets: a(.delay))
            let spatial = crossEntropy(logits: output.spatialLogits, targets: a(.spatial))
            let selectedOffset = takeAlong(output.spatialOffsets, a(.spatial).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let offset = mean(square(selectedOffset - a(.offsets)), axis: -1)
            let selectedArguments = takeAlong(output.continuousArguments, a(.actions).expandedDimensions(axes: [2, 3]), axis: 2).squeezed(axis: 2)
            let arguments = mean(square(selectedArguments - a(.arguments)), axis: -1)
            if balanceInputChoices || actionBalance != nil {
                let choice = actionLoss(logits: output.actionLogits + a(.actionMask), targets: a(.actions), valid: valid,
                                        balanceInputChoices: balanceInputChoices, actionBalance: actionBalance,
                                        sampleWeights: actionBalance == nil ? nil : a(.actionWeights))
                let auxiliary = 0.25 * timing + a(.pointerMask) * (spatial + 2 * offset) + a(.continuousMask) * arguments
                // The same balancing applies to timing and arguments. Otherwise
                // thousands of pointer targets still dominate the shared encoder.
                let weights = actionBalance?.balancesTokens == true
                    ? which(a(.actions) .> 0, a(.actionWeights)[.ellipsis, 1], a(.actionWeights)[.ellipsis, 0])
                    : MLXArray(Float(1))
                loss = choice + sum(auxiliary * weights * valid) / maximum(sum(valid), 1)
            } else {
                // Preserve the existing objective and reduction order for old runs.
                let action = crossEntropy(logits: output.actionLogits + a(.actionMask), targets: a(.actions))
                let perStep = action + 0.25 * timing + a(.pointerMask) * (spatial + 2 * offset) + a(.continuousMask) * arguments
                loss = sum(perStep * valid) / maximum(sum(valid), 1)
            }
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
    var initialPressCorrect: Int?
    var initialPressTotal: Int?
    var accuracy: Double? { total > 0 ? Double(correct) / Double(total) : nil }
    var nonWaitAccuracy: Double? { nonWaitTotal > 0 ? Double(nonWaitCorrect) / Double(nonWaitTotal) : nil }
    var nonWaitPrecision: Double? { nonWaitPredictions > 0 ? Double(nonWaitCorrect) / Double(nonWaitPredictions) : nil }
    var pressRecall: Double? {
        guard let rows = actionBreakdown else { return nil }
        let presses = rows.filter {
            switch $0.action { case .keyDown, .buttonDown: return true; default: return false }
        }
        let count = presses.reduce(0) { $0 + $1.targets }
        return count > 0 ? Double(presses.reduce(0) { $0 + $1.correct }) / Double(count) : nil
    }
    var hasMatchedControl: Bool {
        if let initialPressTotal, initialPressTotal > 0 { return (initialPressCorrect ?? 0) > 0 }
        if let pressRecall { return pressRecall > 0 }
        return nonWaitCorrect > 0
    }
    var collapseWarning: String? {
        guard nonWaitTotal > 0 else { return nil }
        if nonWaitPredictions == 0 {
            return "This checkpoint predicted only Wait despite demonstrated inputs. Its action policy has not learned usable control; review the demonstrations and start a new training run before relying on Run."
        }
        let presses = actionBreakdown?.filter {
            switch $0.action { case .keyDown, .buttonDown: return true; default: return false }
        } ?? []
        if presses.reduce(0, { $0 + $1.targets }) > 0 && presses.reduce(0, { $0 + $1.predictions }) == 0 {
            return "This checkpoint never predicted a key or button press. Correct releases alone cannot start an action in Run."
        }
        if let initialPressTotal, initialPressTotal > 0, initialPressCorrect == 0 {
            return "This checkpoint did not match an initial press from idle. Correct chords and releases do not establish that it can start controlling the task."
        }
        if Double(nonWaitPredictions) > Double(total) * 0.9, Double(nonWaitTotal) < Double(total) * 0.5 {
            return "This checkpoint predicts input on almost every decision although most held-out demonstrations wait. It may overuse controls; verify its behavior in Run before relying on it."
        }
        return nil
    }

    mutating func add(_ other: Self) {
        if total == 0 {
            actionBreakdown = other.actionBreakdown
            initialPressCorrect = other.initialPressCorrect; initialPressTotal = other.initialPressTotal
        } else if other.total > 0 {
            initialPressCorrect = initialPressCorrect.flatMap { own in other.initialPressCorrect.map { own + $0 } }
            initialPressTotal = initialPressTotal.flatMap { own in other.initialPressTotal.map { own + $0 } }
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
                        actions: [ComputerAction], hierarchical: Bool = false, context: MLXArray? = nil) -> Self {
        let prediction = PolicyRunner.selectAction(logits: logits + mask, hierarchical: hierarchical)
        let tokens = MLXArray((0..<actions.count).map(Int32.init))
        let targetColumns = (targets.reshaped([-1, 1]) .== tokens).asType(.float32)
        let predictionColumns = (prediction.reshaped([-1, 1]) .== tokens).asType(.float32)
        let weights = valid.reshaped([-1, 1])
        // Reduce on Metal and read only three counts per vocabulary entry.
        var initialCounts = MLXArray([Float(0), 0])
        if let context {
            let pressTokens: [Float] = actions.map {
                switch $0 { case .keyDown, .buttonDown: return 1; default: return 0 }
            }
            let initial = valid * ((context[.ellipsis, 145] .== 0) & (context[.ellipsis, 146] .== 0)).asType(.float32)
                * take(MLXArray(pressTokens), targets, axis: 0)
            initialCounts = stacked([sum((prediction .== targets).asType(.float32) * initial), sum(initial)])
        }
        let counts = concatenated([stacked([sum(targetColumns * predictionColumns * weights, axis: 0),
                              sum(targetColumns * weights, axis: 0), sum(predictionColumns * weights, axis: 0)]).reshaped([-1]), initialCounts])
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
                    actionBreakdown: rows.filter { $0.targets > 0 || $0.predictions > 0 },
                    initialPressCorrect: context == nil ? nil : Int(counts[3 * n]),
                    initialPressTotal: context == nil ? nil : Int(counts[3 * n + 1]))
    }
}

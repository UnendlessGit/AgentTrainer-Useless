import Foundation
import MLX
import MLXNN
import MLXRandom

struct RunProgress: Sendable {
    var decisions = 0
    var inputTransitions = 0
    var keyPresses: [UInt16: Int] = [:]
    var keyRepeats: [UInt16: Int] = [:]
    var elapsed = 0.0
    var inferenceMilliseconds = 0.0
    var activeMemory = 0
    var cacheMemory = 0
    var lastAction = "Waiting for the first observation"
    var waitingForHuman = false
    var history: [String] = []

    mutating func record(_ action: ComputerAction) {
        decisions += 1
        if case .wait = action {} else { inputTransitions += 1 }
        if case .keyDown(let code) = action { keyPresses[code, default: 0] += 1 }
        if case .keyRepeat(let code) = action { keyRepeats[code, default: 0] += 1 }
    }
}

/// Uses the same preprocessing, codec, grammar and recurrent carry as training.
/// Bounded attention caches frozen visual features, avoiding repeated image encoding.
final class PolicyRunner {
    private let model: PolicyNetwork
    private let codec: PolicyActionCodec
    private let permissions: ActionCapabilities
    private var hidden: [MLXArray] = []
    private var history: [[MLXArray]] = []
    private let instruction: MLXArray

    init(model: PolicyNetwork, permissions: ActionCapabilities, instruction: String) {
        self.model = model; self.permissions = permissions
        codec = PolicyActionCodec(capabilities: model.configuration.capabilities)
        self.instruction = MLXArray(ObservationPreprocessor.instruction(instruction), [1, 1, PolicyNetwork.instructionLength])
    }

    func decide(scene: CapturedScene, state: InputState, previousAction: ComputerAction, elapsed: Double,
                sourceAge: Double, deterministic: Bool, temperature: Float) throws -> PolicyDecision {
        let start = ProcessInfo.processInfo.systemUptime, c = model.configuration
        let image = try ObservationPreprocessor.pixels(scene.image, size: c.imageSize).reshaped([1, 1, c.imageSize, c.imageSize, 3])
        let crop = c.detailCrop ? try ObservationPreprocessor.detailCrop(scene.image, state: state, bounds: scene.bounds, size: c.imageSize)
            .reshaped([1, 1, c.imageSize, c.imageSize, 3]) : MLXArray.zeros([1, 1, c.imageSize, c.imageSize, 3])
        let context = MLXArray(ObservationPreprocessor.context(state: state, bounds: scene.bounds, previousAction: previousAction,
            elapsed: elapsed, sourceAge: sourceAge), [1, 1, PolicyNetwork.contextSize])
        let previous = MLXArray(Int32(codec.token(for: previousAction) ?? codec.count)).reshaped([1, 1])
        var images = image, crops = crop, contexts = context, previousActions = previous
        var encodedVision: MLXArray?, encodedDetail: MLXArray?
        if c.memory == .attention {
            let visual = model.vision(image.reshaped([1, c.imageSize, c.imageSize, 3]))
            let detail = c.detailCrop ? mean(model.vision(crop.reshaped([1, c.imageSize, c.imageSize, 3])), axis: 1) : MLXArray.zeros([1, c.visualWidth])
            eval(visual, detail)
            history.append([context, previous, visual, detail])
            if history.count > c.sequenceLength { history.removeFirst() }
            // Shape-only placeholders remain lazy: encoded visual features are
            // supplied below, so the network never evaluates these image tensors.
            images = MLXArray.zeros([1, history.count, c.imageSize, c.imageSize, 3]); crops = images
            contexts = concatenated(history.map { $0[0] }, axis: 1)
            previousActions = concatenated(history.map { $0[1] }, axis: 1)
            encodedVision = concatenated(history.map { $0[2] }, axis: 0)
            encodedDetail = concatenated(history.map { $0[3] }, axis: 0)
        }
        let length = images.dim(1)
        let output = model(images: images, crops: c.detailCrop ? crops : nil, context: contexts, previousActions: previousActions,
            instructions: broadcast(instruction, to: [1, length, PolicyNetwork.instructionLength]),
            dynamicsActions: MLXArray.zeros([1, length], type: Int32.self), dynamicsArguments: MLXArray.zeros([1, length, 2]),
            hidden: hidden, encodedVision: encodedVision, encodedDetail: encodedDetail)
        let last = length - 1
        let logits = output.actionLogits[0, last] + MLXArray(codec.mask(state: state, capabilities: permissions))
        func choose(_ logits: MLXArray) -> MLXArray {
            deterministic ? argMax(logits, axis: -1) : MLXRandom.categorical(logits / max(0.05, temperature))
        }
        let token = choose(logits)
        let grid = c.imageSize / c.patchSize
        // Patches must intersect the non-letterboxed source rectangle.
        let side = max(scene.bounds.width, scene.bounds.height)
        let left = (1 - scene.bounds.width / side) / 2, top = (1 - scene.bounds.height / side) / 2
        let spatialMask: [Float] = (0..<(grid * grid)).map { index in
            let x = Double(index % grid) / Double(grid), y = Double(index / grid) / Double(grid)
            return x + 1 / Double(grid) > left && x < 1 - left && y + 1 / Double(grid) > top && y < 1 - top ? 0 : -1e9
        }
        let patch = choose(output.spatialLogits[0, last] + MLXArray(spatialMask))
        let offset = take(output.spatialOffsets[0, last], patch, axis: 0)
        let continuous = take(output.continuousArguments[0, last], token, axis: 0)
        let delay = choose(take(output.delayLogits[0, last], token, axis: 0))
        let spatialX = (remainder(patch, grid).asType(.float32) + offset[0]) / Float(grid)
        let spatialY = (floor(patch.asType(.float32) / Float(grid)) + offset[1]) / Float(grid)
        // Gather all selected arguments in one GPU-to-CPU readback.
        let selected = stacked([token.asType(.float32), spatialX, spatialY, continuous[0], continuous[1], delay.asType(.float32)])
        hidden = output.hidden.map { stopGradient($0) }
        eval(selected, hidden)
        let values = selected.asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else { throw DataIntegrityError.invalidData("The model produced non-finite predictions.") }
        let id = Int(values[0]), delayIndex = Int(values[5])
        guard codec.actions.indices.contains(id), PolicyNetwork.delayBins.indices.contains(delayIndex) else {
            throw DataIntegrityError.invalidData("The model produced an invalid action or timing token.")
        }
        let isPointer: Bool
        if case .pointer = codec.actions[id] { isPointer = true } else { isPointer = false }
        let action = try codec.decode(token: id, x: Double(values[isPointer ? 1 : 3]), y: Double(values[isPointer ? 2 : 4]),
            delay: PolicyNetwork.delayBins[delayIndex], bounds: scene.bounds)
        return PolicyDecision(action: action, delay: PolicyNetwork.delayBins[delayIndex],
            inferenceSeconds: ProcessInfo.processInfo.systemUptime - start, observationBounds: scene.bounds)
    }
}

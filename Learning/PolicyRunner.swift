import Foundation
import MLX
import MLXNN
import MLXRandom

struct RunProgress: Sendable {
    var decisions = 0
    var inputTransitions = 0
    var keyPresses: [UInt16: Int] = [:]
    var keyRepeats: [UInt16: Int] = [:]
    var heldInput = InputState()
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
/// Bounded attention caches fused features, avoiding repeated image encoding.
final class PolicyRunner {
    struct MemoryState {
        var hidden: [MLXArray] = []
    }
    private let model: PolicyNetwork
    private let codec: PolicyActionCodec
    private let permissions: ActionCapabilities
    private let hierarchical: Bool
    private let cursorIndependent: Bool
    private(set) var memory = MemoryState()
    private var pendingMemory: MemoryState?
    private let instruction: MLXArray

    init(model: PolicyNetwork, permissions: ActionCapabilities, instruction: String, hierarchical: Bool = false,
         cursorIndependent: Bool = false) throws {
        self.model = model; self.permissions = permissions
        self.hierarchical = hierarchical
        self.cursorIndependent = cursorIndependent
        codec = PolicyActionCodec(capabilities: model.configuration.capabilities)
        self.instruction = MLXArray(try ObservationPreprocessor.instruction(model.configuration.instructionConditioning ? instruction : ""),
                                    [1, 1, PolicyNetwork.instructionLength])
    }

    func decide(scene: CapturedScene, state: InputState, previousAction: ComputerAction, elapsed: Double,
                sourceAge: Double, deterministic: Bool, temperature: Float, commitMemory: Bool = true) throws -> PolicyDecision {
        pendingMemory = nil
        let start = ProcessInfo.processInfo.systemUptime, c = model.configuration
        let image = try ObservationPreprocessor.pixels(scene.image, size: c.imageSize).reshaped([1, 1, c.imageSize, c.imageSize, 3])
        let crop = c.detailCrop && !cursorIndependent ? try ObservationPreprocessor.detailCrop(scene.image, state: state, bounds: scene.bounds, size: c.imageSize)
            .reshaped([1, 1, c.imageSize, c.imageSize, 3]) : MLXArray.zeros([1, 1, c.imageSize, c.imageSize, 3])
        let modelState = ObservationPreprocessor.modelState(state, bounds: scene.bounds, cursorIndependent: cursorIndependent)
        let context = MLXArray(ObservationPreprocessor.context(state: modelState, bounds: scene.bounds, previousAction: previousAction,
            elapsed: elapsed, sourceAge: sourceAge), [1, 1, PolicyNetwork.contextSize])
        let previous = MLXArray(Int32(codec.token(for: previousAction) ?? codec.count)).reshaped([1, 1])
        let output = model(images: image, crops: c.detailCrop ? crop : nil, context: context, previousActions: previous,
            instructions: instruction, dynamicsActions: MLXArray.zeros([1, 1], type: Int32.self),
            dynamicsArguments: MLXArray.zeros([1, 1, 2]), hidden: memory.hidden, wholeSceneDetail: cursorIndependent)
        let last = 0
        let logits = output.actionLogits[0, last] + MLXArray(codec.mask(state: state, capabilities: permissions))
        func choose(_ logits: MLXArray) -> MLXArray {
            deterministic ? argMax(logits, axis: -1) : MLXRandom.categorical(logits / max(0.05, temperature))
        }
        let token = Self.selectAction(logits: logits, hierarchical: hierarchical, deterministic: deterministic, temperature: temperature)
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
        // argMax/categorical return integer indices even when the network emits
        // NaN or infinity. Check the distributions as well as selected arguments.
        let finite = all(isFinite(logits)) & all(isFinite(output.delayLogits[0, last]))
            & all(isFinite(output.spatialLogits[0, last]))
        let selected = stacked([token.asType(.float32), spatialX, spatialY, continuous[0], continuous[1], delay.asType(.float32), finite.asType(.float32)])
        let hidden = output.hidden.map { stopGradient($0) }
        eval(selected, hidden)
        let values = selected.asArray(Float.self)
        guard values.allSatisfy(\.isFinite), values[6] == 1 else { throw DataIntegrityError.invalidData("The model produced non-finite predictions.") }
        let id = Int(values[0]), delayIndex = Int(values[5])
        guard codec.actions.indices.contains(id), PolicyNetwork.delayBins.indices.contains(delayIndex) else {
            throw DataIntegrityError.invalidData("The model produced an invalid action or timing token.")
        }
        let isPointer: Bool
        if case .pointer = codec.actions[id] { isPointer = true } else { isPointer = false }
        let action = try codec.decode(token: id, x: Double(values[isPointer ? 1 : 3]), y: Double(values[isPointer ? 2 : 4]),
            delay: PolicyNetwork.delayBins[delayIndex], bounds: scene.bounds)
        pendingMemory = MemoryState(hidden: hidden)
        if commitMemory { commitDecision() }
        return PolicyDecision(action: action, delay: PolicyNetwork.delayBins[delayIndex],
            inferenceSeconds: ProcessInfo.processInfo.systemUptime - start, observationBounds: scene.bounds)
    }

    /// A decision can become stale while waiting for its predicted timing. Only
    /// actions actually executed advance memory or the bounded attention history.
    func commitDecision() {
        if let pendingMemory { memory = pendingMemory }
        pendingMemory = nil
    }

    func discardDecision() { pendingMemory = nil }

    /// Match the trained wait/input gate, then choose a valid input. Joint argmax
    /// incorrectly compares Wait with each fraction of the total input mass.
    static func selectAction(logits: MLXArray, hierarchical: Bool, deterministic: Bool = true, temperature: Float = 1) -> MLXArray {
        func choose(_ values: MLXArray) -> MLXArray {
            deterministic ? argMax(values, axis: -1) : MLXRandom.categorical(values / max(0.05, temperature))
        }
        guard hierarchical, logits.dim(-1) > 1 else { return choose(logits) }
        let inputs = logits[.ellipsis, 1...]
        let gate = stacked([logits[.ellipsis, 0], logSumExp(inputs, axis: -1)], axis: -1)
        return which(choose(gate) .> 0, choose(inputs) + 1, MLXArray(Int32(0)))
    }
}

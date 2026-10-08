import Foundation
import MLX
import MLXNN
import MLXRandom

final class AttentionBlock: Module {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let attention: MultiHeadAttention
    let expand: Linear
    let project: Linear

    init(width: Int) {
        norm1 = LayerNorm(dimensions: width)
        norm2 = LayerNorm(dimensions: width)
        attention = MultiHeadAttention(dimensions: width, numHeads: 4)
        expand = Linear(width, width * 4)
        project = Linear(width * 4, width)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let n = norm1(x)
        let attended = x + attention(n, keys: n, values: n, mask: mask)
        return attended + project(gelu(expand(norm2(attended))))
    }
}

final class SpatialEncoder: Module {
    let patchProjection: Conv2d
    let position: MLXArray
    let blocks: [AttentionBlock]
    let norm: LayerNorm
    let width: Int
    let gridSize: Int

    init(configuration: PolicyConfiguration) {
        width = configuration.visualWidth
        gridSize = configuration.imageSize / configuration.patchSize
        patchProjection = Conv2d(inputChannels: 3, outputChannels: width,
                                 kernelSize: IntOrPair(configuration.patchSize), stride: IntOrPair(configuration.patchSize))
        position = MLXRandom.normal([1, gridSize * gridSize, width], scale: 0.02)
        blocks = (0..<configuration.visualDepth).map { _ in AttentionBlock(width: configuration.visualWidth) }
        norm = LayerNorm(dimensions: width)
    }
    func callAsFunction(_ images: MLXArray) -> MLXArray {
        var tokens = patchProjection(images).reshaped([images.dim(0), gridSize * gridSize, width]) + position
        for block in blocks { tokens = block(tokens) }
        return norm(tokens)
    }
}

struct PolicyForward {
    var actionLogits: MLXArray
    var spatialLogits: MLXArray
    var spatialOffsets: MLXArray
    var continuousArguments: MLXArray
    var delayLogits: MLXArray
    var futurePixels: MLXArray
    var visualTokens: MLXArray
    var hidden: [MLXArray]
    var temporalFeatures: MLXArray
}

/// One network for arbitrary desktop environments. All spatial tokens survive to
/// the pointer head; a single joint spatial distribution selects a patch and a
/// patch-conditioned residual resolves the precise target within it.
final class PolicyNetwork: Module {
    let vision: SpatialEncoder
    let contextProjection: Linear
    let previousActionEmbedding: Embedding
    let instructionEmbedding: Embedding
    let instructionPosition: MLXArray
    let instructionAttention: AttentionBlock
    let fusion: Linear
    let recurrent: [GRU]
    let temporalAttention: [AttentionBlock]
    let temporalPosition: MLXArray
    let temporalNorm: LayerNorm
    let actionHead: Linear
    let pointerQuery: Linear
    let pointerOffsets: Linear
    let argumentHead: Linear
    let delayHead: Linear
    let dynamicsActionEmbedding: Embedding
    let dynamicsProjection: [Linear]
    let dynamics: Linear
    let configuration: PolicyConfiguration
    let vocabularySize: Int

    static let contextSize = 148
    static let instructionLength = 96
    static let delayBins: [Double] = [0, 0.003, 0.007, 0.012, 0.02, 0.033, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2, 5, 10]

    init(configuration c: PolicyConfiguration) {
        configuration = c
        vocabularySize = PolicyActionCodec(capabilities: c.capabilities).count
        vision = SpatialEncoder(configuration: c)
        contextProjection = Linear(Self.contextSize, 64)
        previousActionEmbedding = Embedding(embeddingCount: vocabularySize + 1, dimensions: 32)
        instructionEmbedding = Embedding(embeddingCount: 257, dimensions: 32)
        instructionPosition = MLXRandom.normal([1, Self.instructionLength, 32], scale: 0.02)
        instructionAttention = AttentionBlock(width: 32)
        fusion = Linear(c.visualWidth * (c.detailCrop ? 2 : 1) + 64 + 32 + (c.instructionConditioning ? 32 : 0), c.memorySize)
        recurrent = c.memory == .recurrent ? (0..<c.memoryDepth).map { _ in GRU(inputSize: c.memorySize, hiddenSize: c.memorySize) } : []
        temporalAttention = c.memory == .attention ? (0..<c.memoryDepth).map { _ in AttentionBlock(width: c.memorySize) } : []
        temporalPosition = MLXRandom.normal([1, c.sequenceLength, c.memorySize], scale: 0.02)
        temporalNorm = LayerNorm(dimensions: c.memorySize)
        actionHead = Linear(c.memorySize, vocabularySize)
        pointerQuery = Linear(c.memorySize, c.visualWidth)
        pointerOffsets = Linear(c.visualWidth + c.memorySize, 2)
        argumentHead = Linear(c.memorySize, vocabularySize * 2)
        delayHead = Linear(c.memorySize, vocabularySize * Self.delayBins.count)
        dynamicsActionEmbedding = Embedding(embeddingCount: vocabularySize + 1, dimensions: 32)
        // Predict low-resolution future RGB values at every spatial patch. A pixel
        // target cannot collapse along with a learned encoder representation.
        let dynamicsWidth = c.visualWidth + c.memorySize + 32 + 2
        dynamicsProjection = c.usesSpatialDynamics ? [Linear(dynamicsWidth, 64)] : []
        dynamics = Linear(c.usesSpatialDynamics ? 64 : dynamicsWidth, 3)
    }

    /// Images: [B,T,H,W,3]. Context/action/instruction semantics are identical in
    /// training and inference. GRU hidden carries are explicit, never global state.
    func callAsFunction(images: MLXArray, crops: MLXArray?, context: MLXArray, previousActions: MLXArray,
                        instructions: MLXArray, dynamicsActions: MLXArray, dynamicsArguments: MLXArray,
                        hidden: [MLXArray] = [], encodedVision: MLXArray? = nil, encodedDetail: MLXArray? = nil,
                        wholeSceneDetail: Bool = false) -> PolicyForward {
        let batch = images.dim(0), length = images.dim(1), n = batch * length
        let c = configuration, patches = vision.gridSize * vision.gridSize
        let tokens = encodedVision ?? vision(images.reshaped([n, c.imageSize, c.imageSize, 3]))
        var features = [mean(tokens, axis: 1).reshaped([batch, length, c.visualWidth]), gelu(contextProjection(context)), previousActionEmbedding(previousActions)]
        if c.detailCrop, let crops {
            let detail = wholeSceneDetail ? mean(tokens, axis: 1)
                : (encodedDetail ?? mean(vision(crops.reshaped([n, c.imageSize, c.imageSize, 3])), axis: 1))
            features.append(detail.reshaped([batch, length, c.visualWidth]))
        }
        if c.instructionConditioning {
            // Instructions are fixed for a recording/run. Encode once per lane,
            // preserving byte order instead of treating language as a bag of bytes.
            let text = instructions[0..., 0, 0...]
            let mask = (text .> 0).asType(.float32)
            let attentionMask = ((1 - mask) * -1e9).expandedDimensions(axes: [1, 2])
            let embedded = instructionAttention(instructionEmbedding(text) + instructionPosition, mask: attentionMask)
            let pooled = sum(embedded * mask.expandedDimensions(axis: -1), axis: 1) / maximum(sum(mask, axis: 1, keepDims: true), 1)
            features.append(broadcast(pooled.expandedDimensions(axis: 1), to: [batch, length, 32]))
        }
        var temporal = gelu(fusion(concatenated(features, axis: -1)))
        var nextHidden: [MLXArray] = []
        if c.memory == .recurrent {
            for (index, layer) in recurrent.enumerated() {
                temporal = layer(temporal, hidden: hidden.indices.contains(index) ? hidden[index] : nil)
                nextHidden.append(temporal[0..., length - 1, 0...])
            }
        } else {
            let result = boundedAttention(temporal, previous: hidden.first)
            temporal = result.output
            nextHidden = [result.history]
        }
        temporal = temporalNorm(temporal)
        let flat = temporal.reshaped([n, c.memorySize])
        let query = pointerQuery(flat).expandedDimensions(axis: 1)
        let spatial = sum(tokens * query, axis: -1) / sqrt(Float(c.visualWidth))
        let broadcastTemporal = broadcast(flat.expandedDimensions(axis: 1), to: [n, patches, c.memorySize])
        let spatialFeatures = concatenated([tokens, broadcastTemporal], axis: -1)
        let offsets = sigmoid(pointerOffsets(spatialFeatures))
        let dynamicsCondition = concatenated([dynamicsActionEmbedding(dynamicsActions), dynamicsArguments], axis: -1)
            .reshaped([n, 34]).expandedDimensions(axis: 1)
        let future = predictFuture(concatenated([spatialFeatures, broadcast(dynamicsCondition, to: [n, patches, 34])], axis: -1))
        return PolicyForward(actionLogits: actionHead(temporal), spatialLogits: spatial.reshaped([batch, length, patches]),
            spatialOffsets: offsets.reshaped([batch, length, patches, 2]),
            continuousArguments: tanh(argumentHead(temporal)).reshaped([batch, length, vocabularySize, 2]),
            delayLogits: delayHead(temporal).reshaped([batch, length, vocabularySize, Self.delayBins.count]),
            futurePixels: future.reshaped([batch, length, patches, 3]), visualTokens: tokens, hidden: nextHidden, temporalFeatures: temporal)
    }

    /// The nonlinear interaction lets one action brighten some patches and
    /// darken others. A single linear projection can only shift every patch in
    /// the same direction within a channel when the action changes.
    func predictFuture(_ features: MLXArray) -> MLXArray {
        let interacted = dynamicsProjection.first.map { gelu($0(features)) } ?? features
        return sigmoid(dynamics(interacted))
    }

    /// Every supervised step sees exactly the trailing window used by Run.
    /// Vision/fusion is evaluated once per new observation; overlapping temporal
    /// windows are batched, with a bounded fused-feature carry across chunks.
    private func boundedAttention(_ current: MLXArray, previous: MLXArray?) -> (output: MLXArray, history: MLXArray) {
        let batch = current.dim(0), length = current.dim(1), width = configuration.memorySize
        let window = configuration.sequenceLength
        let past = previous ?? MLXArray.zeros([batch, 0, width])
        let joined = concatenated([past, current], axis: 1)
        let padded = concatenated([MLXArray.zeros([batch, window - 1, width]), joined], axis: 1)
        let windows = (0..<length).map { index in
            padded[0..., (past.dim(1) + index)..<(past.dim(1) + index + window), 0...]
        }
        var positions: [Int32] = [], paddingMask: [Float] = []
        for index in 0..<length {
            let padding = max(0, window - (past.dim(1) + index + 1))
            for column in 0..<window {
                positions.append(Int32(max(0, column - padding)))
                paddingMask.append(column < padding ? -1e9 : 0)
            }
        }
        let position = take(temporalPosition[0], MLXArray(positions, [length, window]), axis: 0)
        var values = (stacked(windows, axis: 1) + position).reshaped([batch * length, window, width])
        let mask = broadcast(MultiHeadAttention.createAdditiveCausalMask(window)
            + MLXArray(paddingMask, [1, length, 1, 1, window]), to: [batch, length, 1, window, window])
            .reshaped([batch * length, 1, window, window])
        for block in temporalAttention { values = block(values, mask: mask) }
        let retained = min(window - 1, joined.dim(1))
        return (values[0..., window - 1, 0...].reshaped([batch, length, width]),
                joined[0..., (joined.dim(1) - retained)..<joined.dim(1), 0...])
    }
}

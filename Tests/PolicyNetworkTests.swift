import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import AgentTrainer

final class PolicyNetworkTests: XCTestCase {
    func testSpatialDynamicsCanRepresentOppositeLocalizedEffectsOfOneAction() {
        let model = PolicyNetwork(configuration: configuration(.recurrent))
        let width = model.configuration.visualWidth + model.configuration.memorySize + 34
        var projection = Array(repeating: Float(0), count: 64 * width)
        for (row, coefficients) in [[Float(1), 1], [-1, -1], [1, -1], [-1, 1]].enumerated() {
            projection[row * width] = coefficients[0]
            projection[row * width + width - 1] = coefficients[1]
        }
        var output = Array(repeating: Float(0), count: 3 * 64)
        for channel in 0..<3 { for (index, value) in [Float(1), 1, -1, -1].enumerated() { output[channel * 64 + index] = value } }
        model.dynamicsProjection[0].update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray(projection, [64, width]), "bias": MLXArray.zeros([64])]))
        model.dynamics.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray(output, [3, 64]), "bias": MLXArray.zeros([3])]))
        var features = Array(repeating: Float(0), count: 2 * width)
        features[0] = 1; features[width] = -1
        let before = model.predictFuture(MLXArray(features, [1, 2, width]))
        features[width - 1] = 1; features[2 * width - 1] = 1
        let after = model.predictFuture(MLXArray(features, [1, 2, width]))
        XCTAssertEqual(before[0, 0, 0].item(Float.self), 0.5, accuracy: 1e-6)
        XCTAssertEqual(before[0, 1, 0].item(Float.self), 0.5, accuracy: 1e-6)
        XCTAssertGreaterThan(after[0, 0, 0].item(Float.self), 0.7)
        XCTAssertLessThan(after[0, 1, 0].item(Float.self), 0.3)
    }

    func testLegacyDynamicsConfigurationPreservesCheckpointFingerprintAndTensorShape() throws {
        let legacy = Data(#"{"capabilities":{"buttons":[],"chords":false,"dragging":false,"keys":[84,51,83],"maximumHeldKeys":6,"pointer":false,"relativePointer":false,"scrolling":false},"detailCrop":false,"imageSize":128,"instructionConditioning":true,"memory":"Recurrent memory","memoryDepth":2,"memorySize":128,"patchSize":16,"schemaVersion":1,"sequenceLength":32,"visualDepth":3,"visualWidth":64}"#.utf8)
        var c = try JSONDecoder().decode(PolicyConfiguration.self, from: legacy)
        XCTAssertFalse(c.usesSpatialDynamics)
        XCTAssertEqual(c.fingerprint, "1a4f4f6d73b38588014436d95139198ac2fa30b2e32e442e27e5d2d4c7974381")
        let model = PolicyNetwork(configuration: c)
        XCTAssertTrue(model.dynamicsProjection.isEmpty)
        XCTAssertEqual(model.dynamics.weight.shape, [3, 226])
        c.dynamicsPredictor = .linear
        XCTAssertEqual(c.fingerprint, model.configuration.fingerprint)
        c.dynamicsPredictor = .spatialInteraction
        XCTAssertNotEqual(c.fingerprint, model.configuration.fingerprint)
        XCTAssertTrue(PolicyConfiguration().usesSpatialDynamics)
    }

    func testHeldOutMetricsExposeWaitCollapseAndExcludePadding() {
        let logits = MLXArray([Float(9), 1, 0, 9, 1, 0, 9, 1, 0, 9, 1, 0], [1, 4, 3])
        let targets = MLXArray([Int32(0), 1, 2, 0], [1, 4])
        let valid = MLXArray([Float(1), 1, 1, 0], [1, 4])
        let mask = MLXArray([Float(0), 0, 0, 0, 0, 0, -1e9, -1e9, 0, 0, 0, 0], [1, 4, 3])
        let result = ActionEvaluation.measure(logits: logits, targets: targets, valid: valid, mask: mask,
            actions: [.wait(seconds: 0), .keyDown(code: 0), .keyUp(code: 0)])
        XCTAssertEqual(result.total, 3); XCTAssertEqual(result.correct, 2)
        XCTAssertEqual(result.nonWaitTotal, 2); XCTAssertEqual(result.nonWaitCorrect, 1)
        XCTAssertEqual(result.nonWaitPredictions, 1)
        XCTAssertEqual(result.nonWaitAccuracy, 0.5); XCTAssertEqual(result.nonWaitPrecision, 1)
        XCTAssertNil(ActionEvaluation().accuracy)
        XCTAssertEqual(result.actionBreakdown?.map(\.targets), [1, 1, 1])
        XCTAssertEqual(result.actionBreakdown?.map(\.correct), [1, 0, 1])
        XCTAssertEqual(result.actionBreakdown?.map(\.predictions), [2, 0, 1])
        var combined = ActionEvaluation(); combined.add(result); combined.add(result)
        XCTAssertEqual(combined.total, 6)
        XCTAssertEqual(combined.actionBreakdown?.map(\.targets), [2, 2, 2])
        XCTAssertEqual(combined.actionBreakdown?.map(\.correct), [2, 0, 2])
        XCTAssertEqual(combined.actionBreakdown?.map(\.predictions), [4, 0, 2])
    }

    func testLegacyEvaluationDecodesWithoutInventingPerActionCounts() throws {
        let data = Data(#"{"correct":14,"total":20,"nonWaitCorrect":14,"nonWaitTotal":20,"nonWaitPredictions":14}"#.utf8)
        var legacy = try JSONDecoder().decode(ActionEvaluation.self, from: data)
        XCTAssertNil(legacy.actionBreakdown)
        legacy.add(ActionEvaluation(correct: 1, total: 1, actionBreakdown: [
            .init(token: 0, action: .wait(seconds: 0), correct: 1, targets: 1, predictions: 1)]))
        XCTAssertEqual(legacy.total, 21)
        XCTAssertNil(legacy.actionBreakdown)
    }

    func testBalancedChoiceLossPreservesGateFrequencyAndHandlesWaitOnlyBatches() {
        let logits = MLXArray.zeros([1, 3, 3])
        let targets = MLXArray([Int32(0), 1, 2], [1, 3])
        let valid = MLXArray([Float(1), 1, 0], [1, 3])
        let joint = PolicyLoss.actionLoss(logits: logits, targets: targets, valid: valid, balanceInputChoices: false)
        let balanced = PolicyLoss.actionLoss(logits: logits, targets: targets, valid: valid, balanceInputChoices: true)
        XCTAssertEqual(joint.item(Float.self), log(Float(3)), accuracy: 1e-6)
        XCTAssertEqual(balanced.item(Float.self), (log(Float(3)) + log(Float(1.5))) / 2 + log(Float(2)), accuracy: 1e-6)
        let waits = MLXArray.zeros([1, 3], type: Int32.self)
        XCTAssertEqual(PolicyLoss.actionLoss(logits: logits, targets: waits, valid: valid, balanceInputChoices: true).item(Float.self), log(Float(3)), accuracy: 1e-6)
        XCTAssertEqual(PolicyLoss.actionLoss(logits: MLXArray.zeros([1, 3, 1]), targets: waits, valid: valid, balanceInputChoices: true).item(Float.self), 0, accuracy: 1e-6)
        let forced = MLXArray([Float(-1e9), 0, -1e9], [1, 1, 3])
        XCTAssertEqual(PolicyLoss.actionLoss(logits: forced, targets: MLXArray.ones([1, 1], type: Int32.self), valid: MLXArray.ones([1, 1]), balanceInputChoices: true).item(Float.self), 0, accuracy: 1e-6)
    }

    private func configuration(_ architecture: TemporalArchitecture) -> PolicyConfiguration {
        var c = PolicyConfiguration()
        c.imageSize = 128; c.visualWidth = 64; c.visualDepth = 2
        c.memorySize = 128; c.memoryDepth = 1; c.sequenceLength = 4
        c.detailCrop = false; c.instructionConditioning = false; c.memory = architecture
        return c
    }
    private func forward(_ model: PolicyNetwork, images: MLXArray, hidden: [MLXArray] = []) -> PolicyForward {
        let length = images.dim(1)
        return model(images: images, crops: nil,
            context: MLXArray.zeros([1, length, PolicyNetwork.contextSize]),
            previousActions: MLXArray.zeros([1, length], type: Int32.self),
            instructions: MLXArray.zeros([1, length, PolicyNetwork.instructionLength], type: Int32.self),
            dynamicsActions: MLXArray.zeros([1, length], type: Int32.self), dynamicsArguments: MLXArray.zeros([1, length, 2]), hidden: hidden)
    }

    func testRecurrentCarryMatchesFullSequence() {
        let model = PolicyNetwork(configuration: configuration(.recurrent))
        let images = MLXRandom.uniform(0..<1, [1, 4, 128, 128, 3])
        let full = forward(model, images: images)
        let first = forward(model, images: images[0..., 0..<2])
        let second = forward(model, images: images[0..., 2..<4], hidden: first.hidden)
        let error = max(abs(full.actionLogits[0..., 2..<4] - second.actionLogits)).item(Float.self)
        XCTAssertLessThan(error, 1e-4)
        XCTAssertEqual(full.spatialLogits.shape, [1, 4, 64])
        XCTAssertEqual(full.spatialOffsets.shape, [1, 4, 64, 2])
    }

    func testAttentionDoesNotSeeFutureObservations() {
        let model = PolicyNetwork(configuration: configuration(.attention))
        let firstFrames = MLXRandom.uniform(0..<1, [1, 2, 128, 128, 3])
        let a = concatenated([firstFrames, MLXArray.zeros([1, 2, 128, 128, 3])], axis: 1)
        let b = concatenated([firstFrames, MLXArray.ones([1, 2, 128, 128, 3])], axis: 1)
        let outputA = forward(model, images: a)
        let outputB = forward(model, images: b)
        XCTAssertLessThan(max(abs(outputA.actionLogits[0..., 0..<2] - outputB.actionLogits[0..., 0..<2])).item(Float.self), 1e-4)
    }

    func testAttentionCarryMatchesSlidingWindowsAcrossChunksAndDropsOldContext() {
        var c = configuration(.attention); c.memoryDepth = 2
        let model = PolicyNetwork(configuration: c)
        let images = MLXRandom.uniform(0..<1, [1, 9, 128, 128, 3])
        var expected: [MLXArray] = []
        for end in 1...9 {
            let window = forward(model, images: images[0..., max(0, end - c.sequenceLength)..<end])
            expected.append(window.actionLogits[0..., (window.actionLogits.dim(1) - 1)..., 0...])
        }
        let reference = concatenated(expected, axis: 1)
        var carry: [MLXArray] = [], actual: [MLXArray] = []
        for start in stride(from: 0, to: 9, by: 3) {
            let chunk = forward(model, images: images[0..., start..<(start + 3)], hidden: carry)
            carry = chunk.hidden.map { stopGradient($0) }; actual.append(chunk.actionLogits)
            XCTAssertEqual(carry.count, 1)
            XCTAssertEqual(carry[0].shape, [1, c.sequenceLength - 1, c.memorySize])
        }
        XCTAssertLessThan(max(abs(concatenated(actual, axis: 1) - reference)).item(Float.self), 1e-4)
        let changedPrefix = concatenated([MLXArray.ones([1, 5, 128, 128, 3]), images[0..., 5..., 0..., 0..., 0...]], axis: 1)
        let changed = forward(model, images: changedPrefix)
        let original = forward(model, images: images)
        XCTAssertLessThan(max(abs(changed.actionLogits[0, 8] - original.actionLogits[0, 8])).item(Float.self), 1e-4)
    }

    func testDynamicsPretrainingHasVisionGradientsWithoutImitationHeadGradients() {
        let model = PolicyNetwork(configuration: configuration(.recurrent))
        let images = MLXRandom.uniform(0..<1, [1, 2, 128, 128, 3])
        let target = MLXRandom.uniform(0..<1, [1, 2, 64, 3])
        let lossGradient = valueAndGrad(model: model) { [self] model, images, target in
            mean(square(forward(model, images: images).futurePixels - target))
        }
        let (loss, gradients) = lossGradient(model, images, target)
        let values = Dictionary(uniqueKeysWithValues: gradients.flattened())
        XCTAssertTrue(loss.item(Float.self).isFinite)
        let visualGradient = values.filter { $0.key.hasPrefix("vision.") }.values.map { sum(abs($0)).item(Float.self) }.reduce(0, +)
        XCTAssertGreaterThan(visualGradient, 0)
        let imitationGradient = values.filter { $0.key.hasPrefix("actionHead.") }.values.map { sum(abs($0)).item(Float.self) }.reduce(0, +)
        XCTAssertEqual(imitationGradient, 0, accuracy: 1e-8)
    }

    func testOptimizerResumeMatchesUninterruptedUpdate() throws {
        let a = Linear(2, 1), b = Linear(2, 1)
        b.update(parameters: a.parameters())
        let first = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let resumed = ResumableAdamW(learningRate: 0.001, weightDecay: 0.01)
        let x = MLXArray([Float(1), 2, 3, 4], [2, 2])
        let y = MLXArray([Float(2), 4], [2, 1])
        let gradient = valueAndGrad(model: a) { model, x, y in mean(square(model(x) - y)) }
        let (_, g1) = gradient(a, x, y)
        _ = first.update(model: a, gradients: g1, clip: 1)
        b.update(parameters: a.parameters())
        try resumed.restore(first.arrays(), model: b)
        let (_, g2) = gradient(a, x, y)
        _ = first.update(model: a, gradients: g2, clip: 1)
        _ = resumed.update(model: b, gradients: g2, clip: 1)
        XCTAssertLessThan(max(abs(a(x) - b(x))).item(Float.self), 1e-7)
        XCTAssertEqual(first.step, resumed.step)
    }

    func testInstructionOrderChangesPolicyConditioning() {
        var c = configuration(.recurrent); c.instructionConditioning = true
        let model = PolicyNetwork(configuration: c)
        func predict(_ text: String) -> MLXArray {
            model(images: MLXArray.zeros([1, 1, 128, 128, 3]), crops: nil,
                context: MLXArray.zeros([1, 1, PolicyNetwork.contextSize]), previousActions: MLXArray.zeros([1, 1], type: Int32.self),
                instructions: MLXArray(ObservationPreprocessor.instruction(text), [1, 1, PolicyNetwork.instructionLength]),
                dynamicsActions: MLXArray.zeros([1, 1], type: Int32.self), dynamicsArguments: MLXArray.zeros([1, 1, 2])).actionLogits
        }
        XCTAssertGreaterThan(max(abs(predict("open then close") - predict("close then open"))).item(Float.self), 1e-6)
    }
}

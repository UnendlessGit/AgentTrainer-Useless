import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import AgentTrainer

final class PolicyNetworkTests: XCTestCase {
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

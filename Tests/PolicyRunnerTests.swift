import XCTest
import MLX
import MLXNN
import CoreGraphics
@testable import AgentTrainer

final class PolicyRunnerTests: XCTestCase {
    func testRunnerUsesStructuredMaskAndCachedAttentionWithoutInvalidActions() throws {
        for architecture: TemporalArchitecture in [.recurrent, .attention] {
            var c = PolicyConfiguration()
            c.imageSize = 64; c.visualWidth = 32; c.visualDepth = 1; c.memorySize = 32; c.memoryDepth = 1
            c.sequenceLength = 4; c.memory = architecture; c.detailCrop = false; c.instructionConditioning = false
            c.capabilities.keys = [0]; c.capabilities.buttons = []; c.capabilities.scrolling = false; c.capabilities.pointer = false
            let model = PolicyNetwork(configuration: c)
            let runner = try PolicyRunner(model: model, permissions: c.capabilities, instruction: String(repeating: "a", count: 200))
            let reference = try PolicyRunner(model: model, permissions: c.capabilities, instruction: "")
            let provider = try XCTUnwrap(CGDataProvider(data: Data([255, 20, 40, 255]) as CFData))
            let image = try XCTUnwrap(CGImage(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let scene = CapturedScene(image: image, bounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
                sourceTime: 0, availableTime: 10, generation: 1, reusedPixels: false)
            var state = InputState(), previous = ComputerAction.wait(seconds: 0)
            for _ in 0..<6 {
                // A pending decision, invalidated by human input, must leave no
                // extra recurrent step or attention-history entry behind.
                _ = try runner.decide(scene: scene, state: state, previousAction: previous,
                    elapsed: 0.1, sourceAge: 0.1, deterministic: true, temperature: 1, commitMemory: false)
                runner.discardDecision()
                XCTAssertEqual(runner.memory.hidden.count, reference.memory.hidden.count)
                for (actual, expected) in zip(runner.memory.hidden, reference.memory.hidden) {
                    XCTAssertEqual(max(abs(actual - expected)).item(Float.self), 0)
                }
                let decision = try runner.decide(scene: scene, state: state, previousAction: previous,
                    elapsed: 0.1, sourceAge: 0.1, deterministic: true, temperature: 1, commitMemory: false)
                runner.commitDecision()
                let expected = try reference.decide(scene: scene, state: state, previousAction: previous,
                    elapsed: 0.1, sourceAge: 0.1, deterministic: true, temperature: 1)
                XCTAssertEqual(decision.action, expected.action)
                XCTAssertEqual(decision.delay, expected.delay)
                XCTAssertTrue(c.capabilities.permits(decision.action, state: state))
                XCTAssertTrue(decision.delay.isFinite)
                state.apply(decision.action); previous = decision.action
            }
        }
    }
}

extension PolicyRunnerTests {
    func testHierarchicalDecodingKeepsWaitAndInputDecisionsSeparate() {
        // Wait is the largest individual token, but total input probability is 60%.
        let logits = log(MLXArray([Float(0.4), 0.3, 0.3]))
        XCTAssertEqual(PolicyRunner.selectAction(logits: logits, hierarchical: false).item(Int.self), 0)
        XCTAssertEqual(PolicyRunner.selectAction(logits: logits, hierarchical: true).item(Int.self), 1)
        XCTAssertEqual(PolicyRunner.selectAction(logits: MLXArray([Float(2), 0, -1e9]), hierarchical: true).item(Int.self), 0)
        XCTAssertEqual(PolicyRunner.selectAction(logits: MLXArray([Float(0), -1e9, -1e9]), hierarchical: true).item(Int.self), 0)
        XCTAssertEqual(PolicyRunner.selectAction(logits: MLXArray([Float(0)]), hierarchical: true).item(Int.self), 0)
    }

    func testActionBalancingComposesAcrossSparseBatchesAndLegacySettingsDecode() throws {
        let logits = MLXArray.zeros([1, 100, 3])
        let targets = MLXArray(Array(repeating: Int32(0), count: 98) + [1, 2], [1, 100])
        let valid = MLXArray.ones([1, 100])
        let balance = PolicyLoss.ActionBalance(total: 100, inputs: 2)
        let full = PolicyLoss.actionLoss(logits: logits, targets: targets, valid: valid, balanceInputChoices: false, actionBalance: balance)
        let waits = PolicyLoss.actionLoss(logits: logits[0..., ..<98], targets: targets[0..., ..<98], valid: valid[0..., ..<98], balanceInputChoices: false, actionBalance: balance)
        let inputs = PolicyLoss.actionLoss(logits: logits[0..., 98...], targets: targets[0..., 98...], valid: valid[0..., 98...], balanceInputChoices: false, actionBalance: balance)
        XCTAssertEqual(full.item(Float.self), ((98 * waits + 2 * inputs) / 100).item(Float.self), accuracy: 1e-5)
        XCTAssertEqual(balance.waitWeight * 98, balance.inputWeight * 2, accuracy: 1e-5)
        XCTAssertTrue(TrainingSettings().balancesActionFrequency)
        var legacy = TrainingSettings(); legacy.balancedActionFrequency = nil
        let decoded = try JSONDecoder().decode(TrainingSettings.self, from: JSONEncoder().encode(legacy))
        XCTAssertFalse(decoded.balancesActionFrequency)
        let allWait = PolicyLoss.ActionBalance(total: 100, inputs: 0)
        XCTAssertEqual(allWait.waitWeight, 1)
        XCTAssertEqual(allWait.inputWeight, 1)
        let noWait = PolicyLoss.ActionBalance(total: 100, inputs: 100)
        XCTAssertEqual(noWait.inputWeight, 1)
        XCTAssertEqual(noWait.waitWeight, 1)
    }

    func testCollapseDiagnosticsDistinguishLegitimateWaiting() {
        XCTAssertNil(ActionEvaluation(correct: 10, total: 10).collapseWarning)
        XCTAssertNotNil(ActionEvaluation(correct: 98, total: 100, nonWaitTotal: 2).collapseWarning)
        XCTAssertNotNil(ActionEvaluation(correct: 99, total: 100, nonWaitCorrect: 1, nonWaitTotal: 2, nonWaitPredictions: 1,
            actionBreakdown: [.init(token: 1, action: .keyDown(code: 13), correct: 0, targets: 1, predictions: 0),
                              .init(token: 2, action: .keyUp(code: 13), correct: 1, targets: 1, predictions: 1)]).collapseWarning)
    }
}

extension PolicyRunnerTests {
    func testIdlePressesAreBalancedIndependentlyOfFrequentHeldTransitions() {
        let balance = PolicyLoss.ActionBalance(frequencies: [
            ActionFrequency(total: 1000, inputs: 10), ActionFrequency(total: 1000, inputs: 200),
            ActionFrequency(), ActionFrequency()
        ])
        let idle = balance.forState(InputState()), held = balance.forState(InputState(keys: [13]))
        XCTAssertEqual(idle.inputWeight, 50)
        XCTAssertEqual(held.inputWeight, 2.5)
        XCTAssertEqual(idle.inputWeight * 10, idle.waitWeight * 990, accuracy: 1e-4)
        XCTAssertEqual(held.inputWeight * 200, held.waitWeight * 800, accuracy: 1e-4)
        // This state was absent from training. Validation still scores it.
        XCTAssertGreaterThan(balance.forState(InputState(buttons: [0])).inputWeight, 0)
        XCTAssertEqual(ActionFrequency.group(for: InputState(keys: [13], buttons: [0])), 3)
        XCTAssertEqual(ComputerAction.wait(seconds: 1).label(holding: InputState(keys: [13])), "Hold W · 1.000 s")
        XCTAssertEqual(ComputerAction.wait(seconds: 1).label(holding: InputState()), "Wait 1.000 s")
    }
}

extension PolicyRunnerTests {
    func testRunnerRejectsNonFiniteLogitsBeforeEmittingAnAction() throws {
        var c = PolicyConfiguration()
        c.imageSize = 64; c.visualWidth = 32; c.visualDepth = 1; c.memorySize = 32; c.memoryDepth = 1
        c.sequenceLength = 2; c.detailCrop = false; c.instructionConditioning = false
        let model = PolicyNetwork(configuration: c)
        model.actionHead.update(parameters: ModuleParameters.unflattened([
            "bias": MLXArray(Array(repeating: Float.nan, count: model.vocabularySize))
        ]))
        let runner = try PolicyRunner(model: model, permissions: c.capabilities, instruction: "", hierarchical: true)
        let provider = try XCTUnwrap(CGDataProvider(data: Data([0, 0, 0, 255]) as CFData))
        let image = try XCTUnwrap(CGImage(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let scene = CapturedScene(image: image, bounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            sourceTime: 0, availableTime: 1, generation: 1, reusedPixels: false)
        XCTAssertThrowsError(try runner.decide(scene: scene, state: InputState(), previousAction: .wait(seconds: 0),
            elapsed: 0, sourceAge: 0, deterministic: true, temperature: 1))
        XCTAssertTrue(runner.memory.hidden.isEmpty)
    }
}

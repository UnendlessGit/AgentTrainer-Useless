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

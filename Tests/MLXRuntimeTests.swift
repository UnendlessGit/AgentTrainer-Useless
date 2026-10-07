import XCTest
import MLX
import MLXNN
import MLXOptimizers

final class MLXRuntimeTests: XCTestCase {
    func testMetalRuntimeAndGradientUpdate() {
        let model = Linear(2, 1)
        let optimizer = SGD(learningRate: 0.05)
        let x = MLXArray([Float(1), 0, 0, 1, 1, 1, 2, 1], [4, 2])
        let y = MLXArray([Float(1), 2, 3, 4], [4, 1])
        let lossAndGradient = valueAndGrad(model: model) { model, x, y in
            mean(square(model(x) - y))
        }
        let initial = mean(square(model(x) - y)).item(Float.self)
        for _ in 0..<40 {
            let (loss, gradients) = lossAndGradient(model, x, y)
            optimizer.update(model: model, gradients: gradients)
            eval(model, optimizer, loss)
        }
        let final = mean(square(model(x) - y)).item(Float.self)
        XCTAssertTrue(final.isFinite)
        XCTAssertLessThan(final, initial * 0.25)
    }
}

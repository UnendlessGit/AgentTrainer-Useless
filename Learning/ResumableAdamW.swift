import Foundation
import MLX
import MLXNN

/// Explicitly serializable AdamW state; MLX Swift's stock optimizer keeps its
/// per-parameter state private. This implementation retains every moment and
/// the bias-correction step for exact checkpoint continuation.
final class ResumableAdamW {
    var learningRate: Float
    let weightDecay: Float
    private(set) var step = 0
    private var first: [String: MLXArray] = [:]
    private var second: [String: MLXArray] = [:]

    init(learningRate: Float, weightDecay: Float) { self.learningRate = learningRate; self.weightDecay = weightDecay }

    func update(model: Module, gradients: ModuleParameters, clip: Float) -> MLXArray {
        let flattened = gradients.flattened()
        let norm = sqrt(flattened.map { sum(square($0.1)) }.reduce(MLXArray(Float(0)), +))
        let scale = minimum(MLXArray(Float(1)), clip / maximum(norm, 1e-8))
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        step += 1
        let correction1 = Float(1 - pow(0.9, Double(step)))
        let correction2 = Float(1 - pow(0.999, Double(step)))
        var updates: [(String, MLXArray)] = []
        for (key, gradient) in flattened {
            guard let parameter = parameters[key] else { continue }
            let g = gradient * scale
            let m = 0.9 * (first[key] ?? MLXArray.zeros(like: parameter)) + 0.1 * g
            let v = 0.999 * (second[key] ?? MLXArray.zeros(like: parameter)) + 0.001 * square(g)
            first[key] = m; second[key] = v
            // Decay matrices, not biases or normalization scales.
            let decay: Float = parameter.ndim >= 2 ? weightDecay : 0
            let updated = parameter * (1 - learningRate * decay) - learningRate * (m / correction1) / (sqrt(v / correction2) + 1e-8)
            updates.append((key, updated))
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        eval(model, Array(first.values), Array(second.values), norm)
        return norm
    }

    func arrays() -> [String: MLXArray] {
        var values: [String: MLXArray] = ["optimizer_step": MLXArray(Int64(step))]
        for (key, value) in first { values["first." + key] = value }
        for (key, value) in second { values["second." + key] = value }
        return values
    }

    func restore(_ arrays: [String: MLXArray], model: Module) throws {
        guard let stepArray = arrays["optimizer_step"] else { throw DataIntegrityError.invalidData("Optimizer step is missing from this checkpoint.") }
        let restoredStep = stepArray.item(Int.self)
        guard restoredStep >= 0 else { throw DataIntegrityError.invalidData("Invalid optimizer step.") }
        var restoredFirst: [String: MLXArray] = [:], restoredSecond: [String: MLXArray] = [:]
        for (key, parameter) in model.trainableParameters().flattened() {
            guard let m = arrays["first." + key], let v = arrays["second." + key], m.shape == parameter.shape, v.shape == parameter.shape else {
                throw DataIntegrityError.invalidData("Optimizer state does not match parameter \(key).")
            }
            restoredFirst[key] = m; restoredSecond[key] = v
        }
        first = restoredFirst; second = restoredSecond; step = restoredStep
    }
}

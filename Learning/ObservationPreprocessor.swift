import Foundation
import CoreGraphics
import ImageIO
import MLX

/// Shared offline/live preprocessing. Preserve aspect ratio with a centered
/// letterbox, use sRGB, and normalize RGB to [0,1]. Image and pointer transforms
/// must stay in lockstep with PolicyActionCodec.
enum ObservationPreprocessor {
    static let version = 1

    static func image(at url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw DataIntegrityError.invalidData("Could not decode a recorded observation: \(url.lastPathComponent).")
        }
        return image
    }

    static func pixels(_ image: CGImage, size: Int) throws -> MLXArray {
        var data = Data(count: size * size * 4)
        let succeeded = data.withUnsafeMutableBytes { storage -> Bool in
            guard let context = CGContext(data: storage.baseAddress, width: size, height: size, bitsPerComponent: 8,
                bytesPerRow: size * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.interpolationQuality = .high
            context.setFillColor(CGColor(gray: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: size, height: size))
            let scale = Double(size) / Double(max(image.width, image.height))
            let width = Double(image.width) * scale, height = Double(image.height) * scale
            context.draw(image, in: CGRect(x: (Double(size) - width) / 2, y: (Double(size) - height) / 2, width: width, height: height))
            return true
        }
        guard succeeded else { throw DataIntegrityError.io("Could not allocate the model's image input.") }
        return MLXArray(data, [size, size, 4], type: UInt8.self)[0..., 0..., 0..<3].asType(.float32) / 255
    }

    static func detailCrop(_ image: CGImage, state: InputState, bounds: CaptureRect, size: Int) throws -> MLXArray {
        let x = (state.cursorX - bounds.x) / bounds.width * Double(image.width)
        let y = (state.cursorY - bounds.y) / bounds.height * Double(image.height)
        let side = Double(min(size, min(image.width, image.height)))
        let cropRect = CGRect(x: max(0, min(Double(image.width) - side, x - side / 2)),
                              y: max(0, min(Double(image.height) - side, y - side / 2)), width: side, height: side)
        guard let crop = image.cropping(to: cropRect) else { throw DataIntegrityError.invalidData("Could not crop the pointer detail view.") }
        return try pixels(crop, size: size)
    }

    static func context(state: InputState, bounds: CaptureRect, previousAction: ComputerAction,
                        elapsed: Double, sourceAge: Double) -> [Float] {
        var values = [Float](repeating: 0, count: PolicyNetwork.contextSize)
        for key in state.keys where key < 128 { values[Int(key)] = 1 }
        for button in state.buttons where (0..<5).contains(button) { values[128 + button] = 1 }
        values[133] = Float((state.cursorX - bounds.x) / max(1, bounds.width))
        values[134] = Float((state.cursorY - bounds.y) / max(1, bounds.height))
        values[135] = Float(log1p(max(0, elapsed)))
        values[136] = Float(log1p(max(0, sourceAge)))
        values[137] = Float(bounds.width / max(1, bounds.height))
        values[138] = Float(log1p(bounds.width) / 10)
        values[139] = Float(log1p(bounds.height) / 10)
        values[140] = Float(bounds.x / 10_000)
        values[141] = Float(bounds.y / 10_000)
        switch previousAction {
        case .pointer(let x, let y):
            values[142] = Float((x - bounds.x) / max(1, bounds.width)); values[143] = Float((y - bounds.y) / max(1, bounds.height))
        case .relativePointer(let dx, let dy): values[142] = Float(dx / bounds.width); values[143] = Float(dy / bounds.height)
        case .scroll(let dx, let dy): values[142] = Float(dx / 256); values[143] = Float(dy / 256)
        case .wait(let seconds): values[144] = Float(log1p(seconds))
        default: break
        }
        values[145] = state.keys.isEmpty ? 0 : 1
        values[146] = state.buttons.isEmpty ? 0 : 1
        values[147] = (0...1).contains(values[133]) && (0...1).contains(values[134]) ? 1 : 0
        return values.map { $0.isFinite ? min(100, max(-100, $0)) : 0 }
    }

    static func instruction(_ text: String) -> [Int32] {
        let bytes = Array(text.utf8.prefix(PolicyNetwork.instructionLength)).map { Int32($0) + 1 }
        return bytes + Array(repeating: 0, count: PolicyNetwork.instructionLength - bytes.count)
    }
}

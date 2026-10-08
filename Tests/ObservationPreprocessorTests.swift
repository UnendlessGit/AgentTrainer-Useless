import XCTest
import CoreGraphics
import MLX
@testable import AgentTrainer

final class ObservationPreprocessorTests: XCTestCase {
    func testPreservesPixelOrientationAndLetterboxGeometry() throws {
        // Two rows: red/green above blue/white. Raw CGImage bytes are top-first.
        let data = Data([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255])
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let pixels = try ObservationPreprocessor.pixels(image, size: 2).asArray(Float.self)
        XCTAssertEqual(pixels, [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
        let wideImage = try XCTUnwrap(CGImage(width: 2, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let wide = try ObservationPreprocessor.pixels(wideImage, size: 8)
        XCTAssertEqual(sum(wide[0, 0..., 0...]).item(Float.self), 0)
        XCTAssertGreaterThan(sum(wide[3, 0..., 0...]).item(Float.self), 0)
    }

    func testContextAndInstructionHaveStableShapesAndIncludeHeldState() throws {
        var state = InputState(); state.keys = [0, 56]; state.buttons = [0]; state.cursorX = 50; state.cursorY = 20
        let values = ObservationPreprocessor.context(state: state, bounds: CaptureRect(CGRect(x: 0, y: 0, width: 100, height: 100)),
            previousAction: .scroll(dx: 0, dy: 12), elapsed: 0.5, sourceAge: 0.1)
        XCTAssertEqual(values.count, PolicyNetwork.contextSize)
        XCTAssertEqual(values[0], 1); XCTAssertEqual(values[56], 1); XCTAssertEqual(values[128], 1)
        XCTAssertEqual(values[133], 0.5); XCTAssertEqual(values[134], 0.2)
        XCTAssertEqual(try ObservationPreprocessor.instruction("A").first, 66)
        let unicode = String(repeating: "é", count: 48)
        let encoded = try ObservationPreprocessor.instruction(unicode)
        XCTAssertEqual(encoded.count, PolicyNetwork.instructionLength)
        XCTAssertEqual(String(decoding: encoded.map { UInt8($0 - 1) }, as: UTF8.self), unicode)
        XCTAssertThrowsError(try ObservationPreprocessor.instruction(unicode + "a"))
        XCTAssertThrowsError(try ObservationPreprocessor.instruction(String(repeating: "a", count: 200)))
    }
}

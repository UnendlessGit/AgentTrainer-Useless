import XCTest
@testable import AgentTrainer

final class CaptureCatalogTests: XCTestCase {
    func testPixelResolutionUsesRetinaDensityAndKeepsPointGeometryIndependent() throws {
        let window = CGRect(x: 200, y: 190, width: 673, height: 439)
        let retina = try CaptureRasterSize(bounds: window, pixelScale: 2, maximumDimension: 1280)
        XCTAssertEqual(retina.width, 1280)
        XCTAssertEqual(retina.height, 834)
        let native = try CaptureRasterSize(bounds: window, pixelScale: 2, maximumDimension: 2560)
        XCTAssertEqual(native.width, 1346)
        XCTAssertEqual(native.height, 878)
        let ordinary = try CaptureRasterSize(bounds: window, pixelScale: 1, maximumDimension: 2560)
        XCTAssertEqual(ordinary.width, 674)
        XCTAssertEqual(ordinary.height, 440)
        let region = try CaptureRasterSize(bounds: CGRect(x: -500, y: 0, width: 100, height: 80), pixelScale: 2, maximumDimension: 1280)
        XCTAssertEqual(region.width, 200)
        XCTAssertEqual(region.height, 160)
        let oddLimit = try CaptureRasterSize(bounds: window, pixelScale: 2, maximumDimension: 1279)
        XCTAssertLessThanOrEqual(oddLimit.width, 1279)
        XCTAssertEqual(oddLimit.width % 2, 0)
        XCTAssertThrowsError(try CaptureRasterSize(bounds: window, pixelScale: .infinity, maximumDimension: 1280))
        XCTAssertThrowsError(try CaptureRasterSize(bounds: .zero, pixelScale: 2, maximumDimension: 1280))
    }

    @MainActor func testMalformedCaptureRatesAreRejectedBeforeNativeIntegerConversion() throws {
        let catalog = CaptureCatalog()
        for rate in [0, -1, Int.max] {
            var settings = RecordingSettings(); settings.framesPerSecond = rate
            XCTAssertThrowsError(try catalog.resolve(CaptureTarget(), settings: settings)) { error in
                XCTAssertTrue(error.localizedDescription.contains("capture size"))
            }
        }
        var settings = RecordingSettings(); settings.maximumDimension = Int.max
        XCTAssertThrowsError(try settings.validate())
        settings = RecordingSettings(); settings.quality = .nan
        XCTAssertThrowsError(try settings.validate())
        try RecordingSettings().validate()
    }

    @MainActor func testConcurrentRefreshWaitsForTheInFlightResult() async {
        let catalog = CaptureCatalog()
        var release: CheckedContinuation<Void, Never>?
        let first = Task { @MainActor in
            await catalog.refresh {
                await withCheckedContinuation { release = $0 }
                throw DataIntegrityError.io("Refresh fixture completed")
            }
        }
        while release == nil { await Task.yield() }
        var secondEntered = false, secondCompleted = false
        let second = Task { @MainActor in
            secondEntered = true
            await catalog.refresh {
                XCTFail("Concurrent callers should share the existing request")
                throw DataIntegrityError.io("Unexpected second fetch")
            }
            secondCompleted = true
        }
        while !secondEntered { await Task.yield() }
        XCTAssertTrue(catalog.loading)
        XCTAssertFalse(secondCompleted)
        XCTAssertNil(catalog.error)
        release?.resume()
        await first.value; await second.value
        XCTAssertFalse(catalog.loading)
        XCTAssertTrue(secondCompleted)
        XCTAssertEqual(catalog.error, "Refresh fixture completed")
    }
}

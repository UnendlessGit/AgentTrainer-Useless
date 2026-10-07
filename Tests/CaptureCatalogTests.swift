import XCTest
@testable import AgentTrainer

final class CaptureCatalogTests: XCTestCase {
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

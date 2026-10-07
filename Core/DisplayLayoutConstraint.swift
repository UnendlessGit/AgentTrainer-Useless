import Foundation
import CoreGraphics

/// A display capture uses fixed source rectangles. Reusing its buffers after a
/// display moves or disconnects would mislabel pixels with the old coordinates.
struct DisplayLayoutConstraint: Sendable {
    var expected: [UInt32: CGRect]
    var includesEntireDesktop: Bool

    func validate(current: [UInt32: CGRect]) throws {
        if includesEntireDesktop && Set(expected.keys) != Set(current.keys) {
            throw DataIntegrityError.invalidData("The connected displays changed. Refresh capture sources and start again.")
        }
        for (id, bounds) in expected {
            guard current[id] == bounds else {
                throw DataIntegrityError.invalidData("The selected display moved, resized or disconnected. Refresh capture sources and start again.")
            }
        }
    }

    static func current() throws -> [UInt32: CGRect] {
        var count: UInt32 = 0
        guard CGGetActiveDisplayList(0, nil, &count) == .success else {
            throw DataIntegrityError.io("The current display arrangement could not be read.")
        }
        guard count > 0 else { return [:] }
        var ids = [CGDirectDisplayID](repeating: 0, count: Int(count))
        var actualCount = count
        guard CGGetActiveDisplayList(count, &ids, &actualCount) == .success, actualCount <= count else {
            throw DataIntegrityError.io("The display arrangement changed while it was being checked. Refresh capture sources and start again.")
        }
        return Dictionary(uniqueKeysWithValues: ids.prefix(Int(actualCount)).map { ($0, CGDisplayBounds($0)) })
    }
}

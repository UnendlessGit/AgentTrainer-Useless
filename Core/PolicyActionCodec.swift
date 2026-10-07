import Foundation

/// The discrete vocabulary encodes action family AND key/button identity jointly.
/// Spatial arguments are only decoded for the selected movement/scroll family.
struct PolicyActionCodec: Sendable {
    static let version = 1
    let actions: [ComputerAction]

    init(capabilities: ActionCapabilities) {
        var actions: [ComputerAction] = [.wait(seconds: 0)]
        if capabilities.pointer { actions.append(.pointer(x: 0, y: 0)) }
        if capabilities.relativePointer { actions.append(.relativePointer(dx: 0, dy: 0)) }
        if capabilities.scrolling { actions.append(.scroll(dx: 0, dy: 0)) }
        for key in capabilities.keys.sorted() { actions.append(.keyDown(code: key)); actions.append(.keyUp(code: key)) }
        for button in capabilities.buttons.sorted() { actions.append(.buttonDown(button: button)); actions.append(.buttonUp(button: button)) }
        // Capability extension preserves existing configurations' token order,
        // tensor shapes and checkpoint fingerprint.
        if capabilities.repeatsKeys {
            for key in capabilities.keys.sorted() where !(54...63).contains(key) { actions.append(.keyRepeat(code: key)) }
        }
        self.actions = actions
    }

    var count: Int { actions.count }
    func token(for action: ComputerAction) -> Int? { actions.firstIndex { matches($0, action) } }

    func mask(state: InputState, capabilities: ActionCapabilities) -> [Float] {
        actions.map { capabilities.permits($0, state: state) ? 0 : -1e9 }
    }

    func decode(token: Int, x: Double, y: Double, delay: Double, bounds: CaptureRect) throws -> ComputerAction {
        guard actions.indices.contains(token), x.isFinite, y.isFinite, delay.isFinite, bounds.isValid else {
            throw DataIntegrityError.invalidData("The policy produced an invalid action.")
        }
        switch actions[token] {
        case .pointer:
            let side = max(bounds.width, bounds.height)
            return .pointer(x: bounds.x + min(bounds.width - 1, max(0, (x - 0.5) * side + bounds.width / 2)),
                            y: bounds.y + min(bounds.height - 1, max(0, (y - 0.5) * side + bounds.height / 2)))
        case .relativePointer: return .relativePointer(dx: max(-1, min(1, x)) * bounds.width, dy: max(-1, min(1, y)) * bounds.height)
        case .scroll: return .scroll(dx: max(-1, min(1, x)) * 256, dy: max(-1, min(1, y)) * 256)
        case .wait: return .wait(seconds: min(10, max(0, delay)))
        default: return actions[token]
        }
    }

    func arguments(for action: ComputerAction, bounds: CaptureRect) -> (Float, Float) {
        switch action {
        case .pointer(let x, let y):
            let side = max(bounds.width, bounds.height)
            return (Float(0.5 + (x - bounds.x - bounds.width / 2) / side), Float(0.5 + (y - bounds.y - bounds.height / 2) / side))
        case .relativePointer(let dx, let dy): return (Float(dx / bounds.width), Float(dy / bounds.height))
        case .scroll(let dx, let dy): return (Float(dx / 256), Float(dy / 256))
        default: return (0, 0)
        }
    }

    private func matches(_ a: ComputerAction, _ b: ComputerAction) -> Bool {
        switch (a, b) {
        case (.pointer, .pointer), (.relativePointer, .relativePointer), (.scroll, .scroll), (.wait, .wait): return true
        default: return a == b
        }
    }
}

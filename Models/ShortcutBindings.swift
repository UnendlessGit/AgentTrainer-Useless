import Foundation
import Carbon

enum ShortcutAction: UInt32, CaseIterable, Identifiable, Sendable {
    case recording = 1, run = 2, emergency = 3
    var id: UInt32 { rawValue }
    var title: String {
        switch self { case .recording: "Start / stop recording"; case .run: "Start / stop run"; case .emergency: "Emergency stop" }
    }
}

struct ShortcutBinding: Codable, Equatable, Sendable {
    var keyCode: UInt32
    var command = true
    var shift = true
    var option = false
    var control = false
    var carbonModifiers: UInt32 {
        (command ? UInt32(cmdKey) : 0) | (shift ? UInt32(shiftKey) : 0) | (option ? UInt32(optionKey) : 0) | (control ? UInt32(controlKey) : 0)
    }
    var label: String {
        (control ? "⌃" : "") + (option ? "⌥" : "") + (shift ? "⇧" : "") + (command ? "⌘" : "") + KeyNames.name(UInt16(keyCode))
    }
}

struct ShortcutBindings: Codable, Equatable, Sendable {
    var recording = ShortcutBinding(keyCode: 15)
    var run = ShortcutBinding(keyCode: 35)
    var emergency = ShortcutBinding(keyCode: 53, command: true, shift: false, option: true, control: true)
    subscript(_ action: ShortcutAction) -> ShortcutBinding {
        get { switch action { case .recording: recording; case .run: run; case .emergency: emergency } }
        set { switch action { case .recording: recording = newValue; case .run: run = newValue; case .emergency: emergency = newValue } }
    }
    func validate() throws {
        let all = ShortcutAction.allCases.map { self[$0] }
        guard all.allSatisfy({ $0.keyCode < 128 && $0.carbonModifiers != 0 }),
              Set(all.map { "\($0.keyCode)-\($0.carbonModifiers)" }).count == all.count else {
            throw DataIntegrityError.invalidData("Each shortcut needs a modifier and a unique key combination.")
        }
    }
}

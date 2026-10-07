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
    static let supportedKeyCodes: [UInt32] = KeyNames.names.keys
        .filter { !(54...63).contains($0) && ![72, 73, 74].contains($0) }.sorted().map(UInt32.init)
    var keyCode: UInt32
    var command = true
    var shift = true
    var option = false
    var control = false
    var carbonModifiers: UInt32 {
        (command ? UInt32(cmdKey) : 0) | (shift ? UInt32(shiftKey) : 0) | (option ? UInt32(optionKey) : 0) | (control ? UInt32(controlKey) : 0)
    }
    var label: String {
        (control ? "⌃" : "") + (option ? "⌥" : "") + (shift ? "⇧" : "") + (command ? "⌘" : "") + KeyNames.name(UInt16(clamping: keyCode))
    }
}

struct ShortcutBindings: Codable, Equatable, Sendable {
    // Avoid common application shortcuts such as browser reload and command
    // palettes, which remain useful while AgentTrainer is in the background.
    var recording = ShortcutBinding(keyCode: 15, command: true, shift: false, option: true, control: true)
    var run = ShortcutBinding(keyCode: 35, command: true, shift: false, option: true, control: true)
    var emergency = ShortcutBinding(keyCode: 53, command: true, shift: false, option: true, control: true)
    subscript(_ action: ShortcutAction) -> ShortcutBinding {
        get { switch action { case .recording: recording; case .run: run; case .emergency: emergency } }
        set { switch action { case .recording: recording = newValue; case .run: run = newValue; case .emergency: emergency = newValue } }
    }
    func validate() throws {
        let all = ShortcutAction.allCases.map { self[$0] }
        guard all.allSatisfy({ ShortcutBinding.supportedKeyCodes.contains($0.keyCode) && $0.carbonModifiers != 0 }) else {
            throw DataIntegrityError.invalidData("Choose a regular keyboard key and at least one modifier for each shortcut. Modifier and media keys cannot be the shortcut's final key.")
        }
        guard Set(all.map { "\($0.keyCode)-\($0.carbonModifiers)" }).count == all.count else {
            throw DataIntegrityError.invalidData("Each shortcut needs a modifier and a unique key combination.")
        }
    }
}

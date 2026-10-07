import SwiftUI
import AppKit

/// Native checkboxes keep each key's label, state and press action together in
/// the macOS accessibility tree, including inside SwiftUI's lazy grids.
struct KeyPermissionToggle: NSViewRepresentable {
    var key: UInt16
    @Binding var isOn: Bool
    @Environment(\.isEnabled) private var isEnabled

    func makeCoordinator() -> Coordinator { Coordinator(isOn: $isOn) }

    func makeNSView(context: Context) -> NSButton {
        let button = NSButton(checkboxWithTitle: KeyNames.name(key), target: context.coordinator,
                              action: #selector(Coordinator.toggle(_:)))
        button.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return button
    }

    func updateNSView(_ button: NSButton, context: Context) {
        context.coordinator.isOn = $isOn
        button.title = KeyNames.name(key)
        button.setAccessibilityLabel(button.title)
        button.setAccessibilityIdentifier("keyPermission.\(key)")
        button.state = isOn ? .on : .off
        button.isEnabled = isEnabled
    }

    @MainActor final class Coordinator: NSObject {
        var isOn: Binding<Bool>
        init(isOn: Binding<Bool>) { self.isOn = isOn }
        @objc func toggle(_ sender: NSButton) { isOn.wrappedValue = sender.state == .on }
    }
}

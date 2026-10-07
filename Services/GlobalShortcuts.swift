import Foundation
import Carbon

@MainActor
final class GlobalShortcuts {
    private var handler: EventHandlerRef?
    private var registrations: [EventHotKeyRef] = []
    private var bindings: ShortcutBindings?
    var onAction: ((ShortcutAction) -> Void)?

    func install(_ requested: ShortcutBindings) throws {
        try requested.validate()
        if handler == nil {
            var type = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
            let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
                guard let event, let context else { return OSStatus(eventNotHandledErr) }
                var identifier = EventHotKeyID()
                let status = GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                    nil, MemoryLayout<EventHotKeyID>.size, nil, &identifier)
                guard status == noErr, identifier.signature == 0x4154524E, let action = ShortcutAction(rawValue: identifier.id) else { return OSStatus(eventNotHandledErr) }
                let owner = Unmanaged<GlobalShortcuts>.fromOpaque(context).takeUnretainedValue()
                Task { @MainActor in owner.onAction?(action) }
                return noErr
            }, 1, &type, Unmanaged.passUnretained(self).toOpaque(), &handler)
            guard status == noErr else { throw DataIntegrityError.io("Could not install the global shortcut handler (\(status)).") }
        }
        let previous = bindings
        unregister()
        do { try register(requested); bindings = requested }
        catch {
            unregister()
            if let previous { try? register(previous); bindings = previous }
            throw error
        }
    }

    private func register(_ bindings: ShortcutBindings) throws {
        for action in ShortcutAction.allCases {
            let binding = bindings[action]
            var ref: EventHotKeyRef?
            let id = EventHotKeyID(signature: 0x4154524E, id: action.rawValue)
            let status = RegisterEventHotKey(binding.keyCode, binding.carbonModifiers, id, GetApplicationEventTarget(), 0, &ref)
            guard status == noErr, let ref else {
                throw DataIntegrityError.invalidData("\(binding.label) is unavailable or already used by another shortcut. Choose another combination (\(status)).")
            }
            registrations.append(ref)
        }
    }

    private func unregister() { for ref in registrations { UnregisterEventHotKey(ref) }; registrations.removeAll() }
    func stop() {
        unregister()
        if let handler { RemoveEventHandler(handler) }
        handler = nil
    }
}

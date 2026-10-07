import SwiftUI

@main
struct AgentTrainerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var session = AppSession()

    var body: some Scene {
        WindowGroup("AgentTrainer", id: "main") {
            ContentView(session: session)
                .frame(minWidth: 1050, minHeight: 720)
                .preferredColorScheme(session.store.preferences.appearance == "Dark" ? .dark : session.store.preferences.appearance == "Light" ? .light : nil)
                .task { await session.store.load(); session.recorder.permissions.refresh() }
        }
        .defaultSize(width: 1320, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New recording") { session.tab = .record }.keyboardShortcut("n")
            }
            CommandMenu("Session") {
                Button("Stop recording") { Task { await session.recorder.stop() } }
                    .keyboardShortcut("r", modifiers: [.command, .shift]).disabled(!session.recorder.isBusy)
                Button("Emergency stop") { Task { await session.recorder.stop() } }
                    .keyboardShortcut(.escape, modifiers: [.command, .shift])
            }
        }
        Settings { SettingsView(session: session).frame(width: 760, height: 700) }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

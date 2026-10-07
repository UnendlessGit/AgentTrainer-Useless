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
                .task { delegate.session = session; await session.store.load(); session.recorder.permissions.refresh(); session.installShortcuts() }
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

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    var session: AppSession?
    private var terminating = false
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let session else { return .terminateNow }
        session.saveRecordingForm()
        guard session.recorder.isBusy || session.trainer.isBusy || session.store.migrating else { session.shortcuts.stop(); return .terminateNow }
        guard !terminating else { return .terminateLater }
        terminating = true
        session.trainer.pause()
        Task { @MainActor in
            await session.recorder.stop()
            while session.trainer.isBusy || session.store.migrating { try? await Task.sleep(for: .milliseconds(100)) }
            session.shortcuts.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

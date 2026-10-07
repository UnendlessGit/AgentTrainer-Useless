import SwiftUI

struct SettingsView: View {
    var session: AppSession
    @State private var moving = false
    @State private var shortcuts = ShortcutBindings()
    @State private var cleanup: CheckpointCleanupPlan?
    @State private var reviewingCache = false
    @State private var confirmCleanup = false
    private var store: WorkspaceStore { session.store }
    private var permissions: PermissionService { session.recorder.permissions }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(title: "Settings", subtitle: "A workspace that fits your Mac and your workflow.") { EmptyView() }
                Surface(title: "Permissions", symbol: "lock.shield") {
                    permissionRow("Screen Recording", detail: "Capture the display, window or region you select.", granted: permissions.screenRecording,
                                  request: permissions.requestScreenRecording, pane: "Privacy_ScreenCapture")
                    Divider()
                    permissionRow("Input Monitoring", detail: "Record timestamped keyboard and pointer transitions.", granted: permissions.inputMonitoring,
                                  request: permissions.requestInputMonitoring, pane: "Privacy_ListenEvent")
                    Divider()
                    permissionRow("Accessibility", detail: "Execute permitted model actions and stop safely.", granted: permissions.accessibility,
                                  request: permissions.requestAccessibility, pane: "Privacy_Accessibility")
                    Button("Refresh permission status") { permissions.refresh() }
                }
                Surface(title: "Storage", symbol: "externaldrive") {
                    storageRow("Recordings", path: store.preferences.recordingsPath, keyPath: \.recordingsPath)
                    Divider()
                    storageRow("Models", path: store.preferences.modelsPath, keyPath: \.modelsPath)
                    Divider()
                    storageRow("Checkpoints & cache", path: store.preferences.checkpointsPath, keyPath: \.checkpointsPath)
                    Text("Changing a location copies the data first and preserves the original folder. Choose an empty destination.").font(.caption).foregroundStyle(.secondary)
                    if moving { ProgressView("Copying and verifying data…") }
                }
                Surface(title: "Appearance & resources", symbol: "slider.horizontal.3") {
                    Picker("Appearance", selection: preference(\.appearance)) { ForEach(["System", "Light", "Dark"], id: \.self) { Text($0).tag($0) } }.frame(maxWidth: 380)
                    Picker("MLX memory limit", selection: preference(\.memoryLimitGB)) { ForEach([4, 8, 12, 16, 24], id: \.self) { Text("\($0) GB").tag($0) } }.frame(maxWidth: 380)
                    Picker("MLX cache limit", selection: preference(\.cacheLimitGB)) { ForEach([0, 1, 2, 4, 8], id: \.self) { Text("\($0) GB").tag($0) } }.frame(maxWidth: 380)
                    Toggle("Stop runs on human input by default", isOn: preference(\.stopOnHumanInput))
                }
                Surface(title: "Cache & checkpoint storage", symbol: "internaldrive") {
                    Text("Temporary training indexes can be rebuilt. Checkpoint cleanup preserves all latest/best pointers, model references, and three recent checkpoints per stage.")
                        .font(.callout).foregroundStyle(.secondary)
                    HStack {
                        Button("Clear temporary cache") { cleanCache(oldCheckpoints: false) }
                        Button("Review older checkpoints") { Task { await reviewCache() } }
                        if reviewingCache { ProgressView().controlSize(.small) }
                    }.disabled(reviewingCache || !store.activeOperations.isEmpty || store.migrating)
                    if let cleanup {
                        Text("\(cleanup.removable.count) older checkpoints · \(ByteCountFormatter.string(fromByteCount: cleanup.bytes, countStyle: .file)) · \(cleanup.retained) retained")
                            .font(.callout)
                        if !cleanup.issues.isEmpty {
                            Text("Some models were left untouched: \(cleanup.issues.joined(separator: "; "))").font(.caption).foregroundStyle(.orange)
                        }
                        Button("Move older checkpoints to Trash…", role: .destructive) { confirmCleanup = true }
                            .disabled(cleanup.removable.isEmpty || reviewingCache || !store.activeOperations.isEmpty || store.migrating)
                    }
                }
                Surface(title: "Keyboard shortcuts", symbol: "keyboard") {
                    ForEach(ShortcutAction.allCases) { action in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(action.title).frame(width: 170, alignment: .leading)
                                Picker("Key", selection: $shortcuts[action].keyCode) {
                                    ForEach(ShortcutBinding.supportedKeyCodes, id: \.self) { Text(KeyNames.name(UInt16($0))).tag($0) }
                                }.frame(width: 150)
                                Text(shortcuts[action].label).foregroundStyle(.secondary)
                            }
                            HStack {
                                Toggle("Command", isOn: $shortcuts[action].command)
                                Toggle("Shift", isOn: $shortcuts[action].shift)
                                Toggle("Option", isOn: $shortcuts[action].option)
                                Toggle("Control", isOn: $shortcuts[action].control)
                            }.toggleStyle(.checkbox)
                        }
                    }
                    Button("Apply shortcuts") { store.perform { try session.updateShortcuts(shortcuts); store.notice = "Global shortcuts updated." } }
                        .disabled(!store.activeOperations.isEmpty)
                    Text("Available while another app is active. Emergency stop releases agent input when a model is running.").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(30)
        }.task { permissions.refresh(); shortcuts = store.preferences.shortcuts ?? ShortcutBindings() }
        .confirmationDialog("Move older checkpoints to Trash?", isPresented: $confirmCleanup) {
            Button("Move older checkpoints to Trash", role: .destructive) { cleanCache(oldCheckpoints: true) }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Protected checkpoints stay available. Moved checkpoints can be restored from Trash; Finder's Empty Trash reclaims the disk space.") }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in permissions.refresh() }
    }

    private func reviewCache() async {
        reviewingCache = true
        defer { reviewingCache = false }
        do { cleanup = try await store.reviewCheckpointStorage() }
        catch { store.error = error.localizedDescription }
    }

    private func cleanCache(oldCheckpoints: Bool) {
        reviewingCache = true
        Task {
            defer { reviewingCache = false }
            do { try await store.cleanCheckpointStorage(oldCheckpoints: oldCheckpoints); cleanup = try await store.reviewCheckpointStorage() }
            catch { store.error = error.localizedDescription }
        }
    }

    private func preference<T>(_ keyPath: WritableKeyPath<AppPreferences, T>) -> Binding<T> {
        Binding(get: { store.preferences[keyPath: keyPath] }, set: { value in
            var updated = store.preferences; updated[keyPath: keyPath] = value
            store.perform { try store.savePreferences(updated) }
        })
    }
    private func permissionRow(_ title: String, detail: String, granted: Bool, request: @escaping () -> Void, pane: String) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) { Text(title).fontWeight(.medium); Text(detail).font(.caption).foregroundStyle(.secondary) }
            Spacer()
            if granted { Label("Allowed", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.callout) }
            else { Button("Allow", action: request) }
            Button { permissions.openPrivacy(pane) } label: { Image(systemName: "arrow.up.right.square") }.help("Open macOS privacy settings")
        }
    }
    private func storageRow(_ title: String, path: String, keyPath: WritableKeyPath<AppPreferences, String>) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) { Text(title).fontWeight(.medium); Text(path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            Spacer()
            Button("Change…") {
                let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
                panel.message = "Choose an empty folder for \(title.lowercased()). Your existing data will be copied and preserved."
                guard panel.runModal() == .OK, let destination = panel.url else { return }
                moving = true
                Task {
                    do { try await store.relocateStorage(keyPath, to: destination) }
                    catch { store.error = error.localizedDescription }
                    moving = false
                }
            }.disabled(store.migrating || !store.activeOperations.isEmpty)
        }
    }
}

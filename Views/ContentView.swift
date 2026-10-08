import SwiftUI

struct ContentView: View {
    @Bindable var session: AppSession
    var body: some View {
        NavigationSplitView {
            VStack(spacing: 24) {
                HStack(spacing: 10) {
                    Image(systemName: "viewfinder.circle.fill").font(.system(size: 30)).foregroundStyle(.blue.gradient)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("AgentTrainer").font(.headline)
                        Text("Your local learning studio").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                }.padding(.horizontal, 16).padding(.top, 22)
                List(selection: $session.tab) {
                    Section("WORKSPACE") {
                        ForEach(AppTab.allCases.filter { $0 != .settings }) { tab in
                            Label(tab.rawValue, systemImage: tab.symbol).tag(tab).padding(.vertical, 6).accessibilityIdentifier("tab.\(tab.id)")
                        }
                    }
                    Section {
                        Label(AppTab.settings.rawValue, systemImage: AppTab.settings.symbol).tag(AppTab.settings).padding(.vertical, 6)
                    }
                }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 10) {
                    Label("On this Mac", systemImage: "lock.shield").font(.caption.weight(.medium))
                    Text("Recordings, models and training\nstay on your computer.")
                        .font(.caption).foregroundStyle(.secondary).lineSpacing(3)
                    if session.recorder.isBusy { StatusPill(title: session.recorder.phase.rawValue, color: .red) }
                }.frame(maxWidth: .infinity, alignment: .leading).padding(22)
            }.navigationSplitViewColumnWidth(min: 210, ideal: 230, max: 280)
        } detail: {
            Group {
                switch session.tab {
                case .record: RecordView(session: session)
                case .library: LibraryView(store: session.store)
                case .models: ModelsView(store: session.store).disabled(!session.store.activeOperations.isEmpty || session.store.migrating)
                case .train: TrainView(session: session)
                case .run: RunView(session: session)
                case .settings: SettingsView(session: session)
                }
            }
            .disabled(session.store.loading || session.store.workspaceFailure != nil)
            .background(.background.opacity(0.6))
            .overlay {
                if session.store.loading {
                    ProgressView("Loading workspace…").padding(24).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                } else if let failure = session.store.workspaceFailure {
                    VStack(alignment: .leading, spacing: 14) {
                        Label("Workspace unavailable", systemImage: "exclamationmark.triangle").font(.headline)
                        Text(failure).textSelection(.enabled)
                        Button("Retry loading") { Task { await session.loadWorkspace() } }
                            .buttonStyle(.borderedProminent)
                    }.padding(24).frame(maxWidth: 520).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if let notice = session.store.notice {
                    HStack {
                        Label(notice, systemImage: "checkmark.circle.fill").foregroundStyle(.secondary)
                        Spacer()
                        Button { session.store.notice = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain).accessibilityLabel("Dismiss notification")
                    }.font(.callout).padding(14).background(.bar)
                }
            }
        }
        .alert("Something needs attention", isPresented: Binding(get: { session.store.error != nil }, set: { if !$0 { session.store.error = nil } })) {
            Button("OK") { session.store.error = nil }
        } message: { Text(session.store.error ?? "") }
        .toolbar { ToolbarItem { StatusPill(title: "Local workspace", color: .blue) } }
    }
}

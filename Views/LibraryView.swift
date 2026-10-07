import SwiftUI

struct LibraryView: View {
    var store: WorkspaceStore
    @State private var kind: LibraryKind = .imitation
    @State private var folderID: UUID?
    @State private var recordingID: UUID?
    @State private var search = ""
    @State private var folderEditor = false
    @State private var editingFolder: LibraryFolder?
    @State private var folderName = ""
    @State private var parentID: UUID?
    @State private var deleting: RecordingItem?

    private var items: [RecordingItem] {
        store.recordings.filter { $0.manifest.kind == kind && (folderID == nil || $0.manifest.folderID == folderID)
            && (search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) || $0.instruction.localizedCaseInsensitiveContains(search)) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            PageHeader(title: "Library", subtitle: "A growing collection of everything you teach.") {
                Button { editingFolder = nil; folderName = ""; parentID = folderID; folderEditor = true } label: { Label("New folder", systemImage: "folder.badge.plus") }
                    .buttonStyle(.borderedProminent)
            }
            HStack {
                Picker("Data type", selection: $kind) { ForEach(LibraryKind.allCases) { Text($0.rawValue).tag($0) } }.pickerStyle(.segmented).frame(width: 300)
                Spacer()
                TextField("Search recordings", text: $search).textFieldStyle(.roundedBorder).frame(width: 230)
            }
            HSplitView {
                VStack(alignment: .leading, spacing: 10) {
                    Text("FOLDERS").font(.caption.weight(.medium)).foregroundStyle(.secondary).padding(.leading, 10)
                    List(selection: $folderID) {
                        Label("All recordings", systemImage: "square.stack").tag(nil as UUID?)
                        ForEach(store.folders.filter { $0.kind == kind }.sorted { store.folderPath($0) < store.folderPath($1) }) { folder in
                            Label(store.folderPath(folder), systemImage: "folder").tag(Optional(folder.id))
                                .contextMenu {
                                    Button("Rename or move…") { editingFolder = folder; folderName = folder.name; parentID = folder.parentID; folderEditor = true }
                                    Button("Delete empty folder", role: .destructive) { store.perform { try store.deleteFolder(folder) } }
                                }
                        }
                    }.listStyle(.plain)
                }.frame(minWidth: 165, idealWidth: 190, maxWidth: 250)
                VStack(spacing: 0) {
                    if items.isEmpty {
                        EmptyState(symbol: kind == .imitation ? "square.stack.3d.up" : "sparkles.rectangle.stack", title: "Room to learn",
                                   message: kind == .imitation ? "Record a demonstration and it will appear here automatically." : "Save observations in a pre-training folder to teach your model about changes over time.")
                    } else {
                        Table(items, selection: $recordingID) {
                            TableColumn("Recording") { item in
                                HStack(spacing: 10) {
                                    Image(systemName: "play.rectangle").font(.title3).foregroundStyle(.secondary)
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(item.name).fontWeight(.medium)
                                        Text(item.manifest.createdAt, style: .date).font(.caption).foregroundStyle(.secondary)
                                    }
                                }.padding(.vertical, 8)
                            }.width(min: 150, ideal: 240)
                            TableColumn("Duration") { Text(DisplayFormat.duration($0.duration)).monospacedDigit() }.width(70)
                            TableColumn("Frames") { Text($0.manifest.observationCount.formatted()).monospacedDigit() }.width(65)
                            TableColumn("Events") { Text($0.manifest.inputEventCount.formatted()).monospacedDigit() }.width(65)
                            TableColumn("Status") { item in
                                Text(item.eligibility)
                                    .foregroundStyle(item.eligible ? .green : .secondary).font(.caption)
                            }.width(min: 100, ideal: 145)
                        }
                    }
                }.frame(minWidth: 550)
            }
            HStack {
                Text("\(items.count) recordings").foregroundStyle(.secondary)
                Spacer()
                if let item = items.first(where: { $0.id == recordingID }) {
                    Button("Move to Trash…", role: .destructive) { deleting = item }
                        .disabled(!store.activeOperations.isEmpty || store.migrating)
                    Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                    Button("Inspect recording") { inspected = item }.buttonStyle(.borderedProminent).accessibilityLabel("Inspect recording")
                }
            }.font(.callout)
        }.padding(30)
        .confirmationDialog("Move this recording to Trash?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            Button("Move recording to Trash", role: .destructive) {
                if let deleting { store.perform { try store.trashRecording(deleting); recordingID = nil } }
                deleting = nil
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: { Text("The original pixels, input events and edits move together. Models will no longer include this recording in new training runs.") }
        .onChange(of: kind) { _, _ in folderID = nil; recordingID = nil }
        .sheet(item: $inspected) { item in RecordingInspector(store: store, item: item) }
        .sheet(isPresented: $folderEditor) {
            VStack(alignment: .leading, spacing: 20) {
                Text(editingFolder == nil ? "New folder" : "Edit folder").font(.title2.weight(.semibold))
                TextField("Folder name", text: $folderName).textFieldStyle(.roundedBorder)
                Picker("Parent folder", selection: $parentID) {
                    Text("\(kind.rawValue) root").tag(nil as UUID?)
                    ForEach(store.folders.filter { $0.kind == kind && $0.id != editingFolder?.id }) { Text(store.folderPath($0)).tag(Optional($0.id)) }
                }
                HStack {
                    Spacer(); Button("Cancel") { folderEditor = false }.keyboardShortcut(.cancelAction)
                    Button("Save") {
                        do {
                            if var folder = editingFolder { folder.name = folderName; folder.parentID = parentID; try store.updateFolder(folder) }
                            else { folderID = try store.createFolder(name: folderName, kind: kind, parentID: parentID).id }
                            folderEditor = false
                        } catch { folderError = error.localizedDescription }
                    }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                }
                if let folderError { Text(folderError).font(.caption).foregroundStyle(.red) }
            }.padding(28).frame(width: 420)
        }
    }
    @State private var inspected: RecordingItem?
    @State private var folderError: String?
}

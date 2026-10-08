import SwiftUI

/// Keeps large libraries navigable without constructing a control for every file.
struct RecordingAssignmentPicker: View {
    var store: WorkspaceStore
    @Binding var selectedIDs: Set<UUID>
    var folderIDs: Set<UUID>
    @State private var search = ""
    @State private var selectedOnly = false
    @State private var page = 0
    private let pageSize = 20

    var body: some View {
        let folders = Dictionary(store.folders.map { ($0.id, store.folderPath($0)) }, uniquingKeysWith: { first, _ in first })
        let recordings = store.recordings.filter { $0.manifest.kind == .imitation }
        let selectionCount = recordings.filter { selectedIDs.contains($0.id) }.count
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        let matches = recordings.filter { item in
            (!selectedOnly || selectedIDs.contains(item.id)) && (query.isEmpty
                || item.name.localizedCaseInsensitiveContains(query)
                || item.instruction.localizedCaseInsensitiveContains(query)
                || (folders[item.manifest.folderID] ?? "").localizedCaseInsensitiveContains(query))
        }
        let lastPage = max(0, (matches.count - 1) / pageSize)
        let currentPage = min(page, lastPage)
        let start = currentPage * pageSize
        let visible = Array(matches.dropFirst(start).prefix(pageSize))
        VStack(alignment: .leading, spacing: 12) {
            Text("Selected folders already include their recordings. Individual selections add recordings from other folders. Excluded or unreviewed recordings are skipped during training.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Search names, instructions or folders", text: $search)
                .textFieldStyle(.roundedBorder).accessibilityLabel("Search individual recordings")
            Toggle("Only individually selected", isOn: $selectedOnly)
            Text("\(selectionCount) individual \(selectionCount == 1 ? "selection" : "selections")")
                .font(.caption).foregroundStyle(.secondary)
            if matches.isEmpty {
                Text(selectedOnly ? "No individual selections match this search." : "No recordings match this search.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(visible) { item in
                let includedByFolder = folderIDs.contains(item.manifest.folderID)
                Toggle(isOn: Binding(get: { includedByFolder || selectedIDs.contains(item.id) }, set: { included in
                    if included { selectedIDs.insert(item.id) } else { selectedIDs.remove(item.id) }
                })) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(item.name)
                        Text("\(folders[item.manifest.folderID] ?? "Unfiled") · \(DisplayFormat.duration(item.duration)) · \(item.manifest.createdAt.formatted(date: .abbreviated, time: .standard))")
                            .font(.caption).foregroundStyle(.secondary)
                        if includedByFolder { Text("Included by selected folder").font(.caption).foregroundStyle(.secondary) }
                        if !item.eligible { Text("Not used for training: \(item.eligibility)").font(.caption).foregroundStyle(.orange) }
                    }
                }
                .disabled(includedByFolder)
                .accessibilityIdentifier("assignment.\(item.id.uuidString)")
                .help("Recording \(item.id.uuidString)")
            }
            if !matches.isEmpty {
                HStack {
                    Text(matches.count == 1 ? "1 recording" : "\(start + 1)–\(start + visible.count) of \(matches.count) recordings")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Previous") { page = currentPage - 1 }.disabled(currentPage == 0)
                    Button("Next") { page = currentPage + 1 }.disabled(currentPage == lastPage)
                }
            }
        }
        .onChange(of: search) { _, _ in page = 0 }
        .onChange(of: selectedOnly) { _, _ in page = 0 }
    }
}

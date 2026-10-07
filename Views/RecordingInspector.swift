import SwiftUI

struct RecordingInspector: View {
    var store: WorkspaceStore
    let item: RecordingItem
    @Environment(\.dismiss) private var dismiss
    @State private var observations: [VisualObservation] = []
    @State private var events: [InputTransition] = []
    @State private var frame = 0.0
    @State private var name = ""
    @State private var instruction = ""
    @State private var trimStart = 0.0
    @State private var trimEnd = 0.0
    @State private var excluded = false
    @State private var folderID: UUID?
    @State private var error: String?
    @State private var loading = true
    @State private var image: NSImage?

    private var current: VisualObservation? { observations.indices.contains(Int(frame)) ? observations[Int(frame)] : nil }
    private var nearbyEvents: [InputTransition] {
        guard let current else { return Array(events.prefix(200)) }
        let start = current.timeNanoseconds
        let end = Int(frame) + 1 < observations.count ? observations[Int(frame) + 1].timeNanoseconds : UInt64.max
        return events.filter { $0.timeNanoseconds > start && $0.timeNanoseconds <= end }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Inspect recording").font(.title2.weight(.semibold)); Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            HSplitView {
                VStack(spacing: 12) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 12).fill(.black.opacity(0.9))
                        if let image { Image(nsImage: image).resizable().scaledToFit() }
                        else if loading { ProgressView() }
                        else { Text("No preview available").foregroundStyle(.white.opacity(0.6)) }
                    }.frame(minHeight: 230, maxHeight: 360)
                    Slider(value: $frame, in: 0...Double(max(1, observations.count - 1)), step: 1).disabled(observations.count < 2)
                    HStack {
                        Text("Observation \(min(Int(frame) + 1, observations.count)) / \(item.manifest.observationCount)")
                        Spacer()
                        Text(String(format: "%.3f s", Double(current?.timeNanoseconds ?? 0) / 1e9)).monospacedDigit()
                    }.font(.caption).foregroundStyle(.secondary)
                    HStack { Text("Actions after this observation").font(.headline); Spacer(); Text("\(nearbyEvents.count) events").foregroundStyle(.secondary) }
                    List(nearbyEvents) { event in
                        HStack {
                            Text(String(format: "%.6f s", Double(event.timeNanoseconds) / 1e9)).monospaced().foregroundStyle(.secondary)
                            Text(event.action.label)
                            if event.isRepeat { Text("repeat").font(.caption).foregroundStyle(.secondary) }
                        }
                    }.frame(minHeight: 160)
                    if item.manifest.observationCount > observations.count || item.manifest.inputEventCount > events.count {
                        Text("Preview is limited to 10,000 observations and 100,000 input events. Original journals remain complete.").font(.caption).foregroundStyle(.secondary)
                    }
                }.frame(minWidth: 500).padding(.trailing, 16)
                Form {
                    Section("Metadata") { TextField("Name", text: $name); TextField("Instruction", text: $instruction, axis: .vertical) }
                    Section("Nondestructive trim") {
                        if let reason = item.edits.automaticTrimReason { Text(reason).font(.caption).foregroundStyle(.secondary) }
                        TextField("Start (seconds)", value: $trimStart, format: .number)
                        TextField("End (seconds)", value: $trimEnd, format: .number)
                        Text("Original pixels and events are preserved.").font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Library") {
                        Picker("Folder", selection: $folderID) {
                            ForEach(store.folders.filter { $0.kind == item.manifest.kind }) { Text(store.folderPath($0)).tag(Optional($0.id)) }
                        }
                        Toggle("Exclude from training", isOn: $excluded)
                    }
                    Section {
                        LabeledContent("Recorded", value: item.manifest.createdAt.formatted())
                        LabeledContent("Input events", value: item.manifest.inputEventCount.formatted())
                        LabeledContent("Status", value: item.manifest.eligibility)
                    }
                    Button("Save changes") { save() }.buttonStyle(.borderedProminent)
                    if let error { Text(error).foregroundStyle(.red).font(.caption) }
                }.formStyle(.grouped).frame(width: 320)
            }
        }.padding(26).frame(width: 950, height: 740)
        .task {
            name = item.name; instruction = item.instruction; trimStart = item.edits.trimStart
            trimEnd = item.edits.trimEnd ?? item.manifest.duration; excluded = item.edits.excluded; folderID = item.manifest.folderID
            do {
                let url = item.url
                let result = try await Task.detached {
                    let observations = try JSONLines.load(VisualObservation.self, from: url.appendingPathComponent("observations.jsonl"))
                    let events = try JSONLines.load(InputTransition.self, from: url.appendingPathComponent("events.jsonl"), limit: 100_000)
                    return (observations, events)
                }.value
                observations = result.0; events = result.1; updateImage()
            } catch { self.error = error.localizedDescription }
            loading = false
        }
        .onChange(of: frame) { _, _ in updateImage() }
    }
    private func updateImage() {
        guard let current, RecordingJournal.isSafeFramePath(current.imageFile) else { image = nil; return }
        image = NSImage(contentsOf: item.url.appendingPathComponent(current.imageFile))
    }
    private func save() {
        do {
            let edits = RecordingEdits(name: name, instruction: instruction, trimStart: trimStart, trimEnd: trimEnd, excluded: excluded)
            try store.editRecording(item, edits: edits)
            if let folderID, folderID != item.manifest.folderID { try store.moveRecording(item, to: folderID) }
            store.notice = "Recording changes saved. Original data preserved."
            dismiss()
        } catch { self.error = error.localizedDescription }
    }
}

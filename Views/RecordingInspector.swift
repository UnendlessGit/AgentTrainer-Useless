import SwiftUI

struct RecordingInspector: View {
    var store: WorkspaceStore
    let item: RecordingItem
    @Environment(\.dismiss) private var dismiss
    @State private var preview: RecordingPreview?
    @State private var currentFrame: RecordingPreviewFrame?
    @State private var frame = 0.0
    @State private var name = ""
    @State private var instruction = ""
    @State private var trimStart = 0.0
    @State private var trimEnd = 0.0
    @State private var excluded = false
    @State private var reviewedRecovery = false
    @State private var folderID: UUID?
    @State private var error: String?
    @State private var loading = true
    private var observationCount: Int { preview?.observations.count ?? 0 }

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
                        if let currentFrame { Image(decorative: currentFrame.image, scale: 1).resizable().scaledToFit() }
                        else if loading { ProgressView() }
                        else { Text("No preview available").foregroundStyle(.white.opacity(0.6)) }
                    }.frame(minHeight: 230, maxHeight: 360)
                    HStack {
                        Button { frame = max(0, frame - 1) } label: { Image(systemName: "backward.frame") }
                            .accessibilityLabel("Previous observation").disabled(frame <= 0)
                        Slider(value: $frame, in: 0...Double(max(1, observationCount - 1)), step: 1)
                            .accessibilityLabel("Observation timeline").disabled(observationCount < 2)
                        Button { frame = min(Double(observationCount - 1), frame + 1) } label: { Image(systemName: "forward.frame") }
                            .accessibilityLabel("Next observation").disabled(Int(frame) >= observationCount - 1)
                    }
                    HStack {
                        Text("Observation \(min(Int(frame) + 1, observationCount)) / \(observationCount)")
                        Spacer()
                        Text(String(format: "%.3f s", Double(currentFrame?.observation.timeNanoseconds ?? 0) / 1e9)).monospacedDigit()
                    }.font(.caption).foregroundStyle(.secondary)
                    HStack { Text("Actions after this observation").font(.headline); Spacer(); Text("\(currentFrame?.eventCount ?? 0) events").foregroundStyle(.secondary) }
                    List(currentFrame?.events ?? []) { event in
                        HStack {
                            Text(String(format: "%.6f s", Double(event.timeNanoseconds) / 1e9)).monospaced().foregroundStyle(.secondary)
                            Text(event.action.label)
                            if event.isRepeat { Text("repeat").font(.caption).foregroundStyle(.secondary) }
                        }
                    }.frame(minHeight: 160)
                    if let currentFrame, currentFrame.eventCount > currentFrame.events.count {
                        Text("Showing the first 1,000 events in this observation interval. All events remain in the original journal.").font(.caption).foregroundStyle(.secondary)
                    }
                }.frame(minWidth: 500).padding(.trailing, 16)
                Form {
                    Section("Metadata") {
                        TextField("Name", text: $name)
                        TextField("Instruction", text: $instruction, axis: .vertical)
                        InstructionSizeHint(text: instruction)
                    }
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
                        if item.needsRecoveryReview {
                            Text(item.manifest.failure ?? "Recording was interrupted.").font(.caption).foregroundStyle(.secondary)
                            Toggle("I reviewed this recovered recording", isOn: $reviewedRecovery)
                            Text("Only complete journal entries are used. Trim incomplete work before enabling training.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Section {
                        LabeledContent("Recorded", value: item.manifest.createdAt.formatted())
                        LabeledContent("Input events", value: item.manifest.inputEventCount.formatted())
                        if let currentFrame {
                            LabeledContent("Image size", value: "\(currentFrame.observation.width) × \(currentFrame.observation.height) px")
                        }
                        LabeledContent("Requested rate", value: "\(item.manifest.settings.framesPerSecond) observations/s")
                        if item.manifest.duration > 0 {
                            LabeledContent("Session average", value: String(format: "%.1f observations/s", Double(item.manifest.observationCount) / item.manifest.duration))
                        }
                        LabeledContent("Frames replaced", value: item.manifest.droppedVisualFrames.formatted())
                            .help("Received frames replaced before sampling. This count excludes frames macOS did not deliver. Input transitions use a separate journal.")
                        LabeledContent("Status", value: item.eligibility)
                    }
                    Button("Save changes") { save() }.buttonStyle(.borderedProminent)
                        .disabled(!store.activeOperations.isEmpty || store.migrating)
                    if let error { Text(error).foregroundStyle(.red).font(.caption) }
                }.formStyle(.grouped).frame(width: 320)
            }
        }.padding(26).frame(width: 950, height: 740)
        .task {
            name = item.name; instruction = item.instruction; trimStart = item.edits.trimStart
            trimEnd = item.edits.trimEnd ?? item.manifest.duration; excluded = item.edits.excluded; folderID = item.manifest.folderID
            reviewedRecovery = item.edits.reviewedRecovery ?? false
            do {
                let url = item.url
                let worker = Task.detached(priority: .userInitiated) { try RecordingPreview(recording: url) }
                let result = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                try Task.checkCancellation()
                preview = result
            } catch is CancellationError { return }
            catch { self.error = error.localizedDescription }
            loading = false
        }
        .task(id: "\(observationCount):\(Int(frame))") {
            guard let preview else { return }
            let selected = Int(frame)
            do {
                let worker = Task.detached(priority: .userInitiated) { try preview.frame(at: selected) }
                let result = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                try Task.checkCancellation()
                currentFrame = result
            } catch is CancellationError { /* A newer scrub position superseded this frame. */ }
            catch { self.error = error.localizedDescription }
        }
    }
    private func save() {
        do {
            var edits = item.edits
            edits.name = name; edits.instruction = instruction; edits.trimStart = trimStart; edits.trimEnd = trimEnd
            edits.excluded = excluded; edits.reviewedRecovery = reviewedRecovery
            try store.editRecording(item, edits: edits)
            if let folderID, folderID != item.manifest.folderID { try store.moveRecording(item, to: folderID) }
            store.notice = "Recording changes saved. Original data preserved."
            dismiss()
        } catch { self.error = error.localizedDescription }
    }
}

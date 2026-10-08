import SwiftUI

struct ModelsView: View {
    var store: WorkspaceStore
    @State private var selectedID: UUID?
    @State private var creating = false
    @State private var name = "My first agent"
    @State private var deleting: AIModel?
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            PageHeader(title: "AI Models", subtitle: "Shape what your agents see, remember and do.") {
                Button { creating = true } label: { Label("Create model", systemImage: "plus") }.buttonStyle(.borderedProminent).accessibilityIdentifier("models.create")
            }
            if store.models.isEmpty {
                EmptyState(symbol: "cpu", title: "An agent of your own", message: "Create a model, choose its capabilities, and connect the demonstrations it will learn from.")
            } else {
                HSplitView {
                    List(selection: $selectedID) {
                        ForEach(store.models) { model in
                            VStack(alignment: .leading, spacing: 7) {
                                Label(model.name, systemImage: "cpu").font(.headline)
                                Text(model.compatibility).font(.caption).foregroundStyle(.secondary)
                            }.padding(.vertical, 10).tag(Optional(model.id))
                            .contextMenu {
                                Button("Duplicate configuration") { duplicate(model) }
                                Button("Move model to Trash…", role: .destructive) { deleting = model }
                            }
                        }
                    }.frame(minWidth: 180, idealWidth: 230, maxWidth: 300)
                    if let model = store.models.first(where: { $0.id == selectedID }) {
                        ModelEditor(store: store, original: model).id(model.id)
                    } else { EmptyState(symbol: "cpu", title: "Select a model", message: "Review its architecture, capabilities and training data.") }
                }
                if let model = store.models.first(where: { $0.id == selectedID }) {
                    HStack {
                        Button("Duplicate configuration") { duplicate(model) }.accessibilityLabel("Duplicate model configuration")
                        Spacer()
                        Button("Move model to Trash…", role: .destructive) { deleting = model }.accessibilityLabel("Move model to Trash")
                    }
                }
            }
        }.padding(30)
        .confirmationDialog("Move this model to Trash?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            Button("Move model to Trash", role: .destructive) {
                if let deleting { store.perform { try store.trashModel(deleting); selectedID = store.models.first?.id } }
                deleting = nil
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: { Text("The configuration can be restored from Trash. Its checkpoints are preserved in checkpoint storage.") }
        .task { if selectedID == nil { selectedID = store.models.first?.id } }
        .sheet(isPresented: $creating) {
            VStack(alignment: .leading, spacing: 20) {
                Text("Create a model").font(.title2.weight(.semibold))
                Text("Start with a spatial vision encoder and temporal memory. You can configure architecture and capabilities before training.").foregroundStyle(.secondary)
                TextField("Model name", text: $name).textFieldStyle(.roundedBorder).accessibilityIdentifier("models.name")
                HStack { Spacer(); Button("Cancel") { creating = false }; Button("Create model") {
                    do { selectedID = try store.createModel(name: name).id; creating = false }
                    catch { store.error = error.localizedDescription }
                }.buttonStyle(.borderedProminent).disabled(name.trimmingCharacters(in: .whitespaces).isEmpty).accessibilityIdentifier("models.confirmCreate") }
            }.padding(28).frame(width: 440)
        }
    }

    private func duplicate(_ model: AIModel) {
        store.perform { selectedID = try store.createModel(name: model.name + " copy", copying: model).id }
    }
}

private struct ModelEditor: View {
    var store: WorkspaceStore
    let original: AIModel
    @State private var draft: AIModel
    @State private var showKeys = false
    @State private var showRecordings = false
    init(store: WorkspaceStore, original: AIModel) {
        self.store = store; self.original = original
        _draft = State(initialValue: store.modelDrafts[original.id] ?? original)
    }
    private var saved: AIModel { store.models.first(where: { $0.id == original.id }) ?? original }
    private var hasUnsavedChanges: Bool {
        draft.name != saved.name || draft.configuration != saved.configuration
            || draft.imitationFolderIDs != saved.imitationFolderIDs
            || draft.imitationRecordingIDs != saved.imitationRecordingIDs
            || draft.pretrainingFolderIDs != saved.pretrainingFolderIDs
    }
    private var checkpointCompatibility: String {
        var current = store.models.first(where: { $0.id == original.id }) ?? original
        current.configuration = draft.configuration
        return current.compatibility
    }
    var body: some View {
        Form {
            Section("Model") {
                TextField("Name", text: $draft.name)
                LabeledContent("Checkpoint", value: checkpointCompatibility)
                if draft.configuration.fingerprint != saved.configuration.fingerprint {
                    Label("Architecture or capability changes require retraining.", systemImage: "arrow.triangle.2.circlepath").foregroundStyle(.orange)
                }
            }
            Section("Vision") {
                LabeledContent("Encoder", value: "Spatial patch encoder · trained locally")
                Picker("Input resolution", selection: $draft.configuration.imageSize) { ForEach([128, 224, 320, 448], id: \.self) { Text("\($0) × \($0)").tag($0) } }
                Picker("Visual width", selection: $draft.configuration.visualWidth) { ForEach([64, 128, 256], id: \.self) { Text("\($0)").tag($0) } }
                Stepper("Visual layers: \(draft.configuration.visualDepth)", value: $draft.configuration.visualDepth, in: 2...6)
                Toggle("High-detail cursor crop", isOn: $draft.configuration.detailCrop)
            }
            Section("Memory & context") {
                Picker("Temporal architecture", selection: $draft.configuration.memory) { ForEach(TemporalArchitecture.allCases) { Text($0.rawValue).tag($0) } }
                Picker("Memory width", selection: $draft.configuration.memorySize) { ForEach([128, 256, 512], id: \.self) { Text("\($0)").tag($0) } }
                Stepper("Memory layers: \(draft.configuration.memoryDepth)", value: $draft.configuration.memoryDepth, in: 1...4)
                Picker("Sequence length", selection: $draft.configuration.sequenceLength) { ForEach([16, 32, 64, 128], id: \.self) { Text("\($0) decisions").tag($0) } }
                Toggle("Use task instructions", isOn: $draft.configuration.instructionConditioning)
                Text("Memory advances with each input transition or wait decision. Geometry, cursor position, previous actions, elapsed time and input state are included.").font(.caption).foregroundStyle(.secondary)
            }
            Section("Action capabilities") {
                Toggle("Move pointer", isOn: $draft.configuration.capabilities.pointer)
                Toggle("Relative pointer movement", isOn: $draft.configuration.capabilities.relativePointer)
                Toggle("Scrolling", isOn: $draft.configuration.capabilities.scrolling)
                Toggle("Dragging", isOn: $draft.configuration.capabilities.dragging)
                Toggle("Key combinations", isOn: $draft.configuration.capabilities.chords)
                Stepper("Maximum held keys: \(draft.configuration.capabilities.maximumHeldKeys)",
                        value: $draft.configuration.capabilities.maximumHeldKeys, in: 1...16)
                Toggle("Repeated key events while held", isOn: $draft.configuration.capabilities.repeatsKeys)
                ForEach(ActionCapabilities.supportedButtons, id: \.self) { code in
                    Toggle(KeyNames.button(code), isOn: Binding(get: { draft.configuration.capabilities.buttons.contains(code) }, set: {
                        if $0 { draft.configuration.capabilities.buttons.insert(code) } else { draft.configuration.capabilities.buttons.remove(code) }
                    }))
                }
                Button("\(showKeys ? "Hide" : "Choose") keyboard keys · \(draft.configuration.capabilities.keys.count) allowed") { showKeys.toggle() }
                    .accessibilityLabel("Choose allowed keyboard keys").accessibilityIdentifier("models.chooseKeys")
                Text("Enable the controls used by your demonstrations. Held keys stay down between transitions; a release ends the hold. More allowed keys increase the choices the policy must learn.")
                    .font(.caption).foregroundStyle(.secondary)
                if showKeys {
                    HStack {
                        Button("All listed keys") { draft.configuration.capabilities.keys = Set(KeyNames.names.keys) }
                        Button("Clear keys") { draft.configuration.capabilities.keys.removeAll() }
                    }
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 105))], alignment: .leading, spacing: 8) {
                        ForEach(KeyNames.names.keys.sorted(), id: \.self) { code in
                            KeyPermissionToggle(key: code, isOn: Binding(get: { draft.configuration.capabilities.keys.contains(code) }, set: {
                                if $0 { draft.configuration.capabilities.keys.insert(code) } else { draft.configuration.capabilities.keys.remove(code) }
                            }))
                        }
                    }.padding(.vertical, 10)
                }
            }
            Section("Imitation-learning data") {
                Text("Demonstrations teach the agent which actions to take.").font(.caption).foregroundStyle(.secondary)
                ForEach(store.folders.filter { $0.kind == .imitation }) { folder in
                    Toggle(store.folderPath(folder), isOn: Binding(get: { draft.imitationFolderIDs.contains(folder.id) }, set: {
                        if $0 { draft.imitationFolderIDs.insert(folder.id) } else { draft.imitationFolderIDs.remove(folder.id) }
                    }))
                }
                Button(showRecordings ? "Hide individual recordings" : "Choose individual recordings") { showRecordings.toggle() }
                    .accessibilityLabel("Choose individual recordings")
                if showRecordings {
                    RecordingAssignmentPicker(store: store, selectedIDs: $draft.imitationRecordingIDs,
                                              folderIDs: draft.imitationFolderIDs)
                }
            }
            Section("Pre-training data") {
                Picker("Visual dynamics predictor", selection: Binding(
                    get: { draft.configuration.dynamicsPredictor ?? .linear },
                    set: { draft.configuration.dynamicsPredictor = $0 }
                )) {
                    ForEach(DynamicsPredictor.allCases) { Text($0.rawValue).tag($0) }
                }
                Text("Spatial interaction learns action effects that differ by image location. Changing this architecture requires retraining; it does not guarantee better imitation.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Separate observations for learning temporal and action-conditioned representations before imitation learning.").font(.caption).foregroundStyle(.secondary)
                ForEach(store.folders.filter { $0.kind == .pretraining }) { folder in
                    Toggle(store.folderPath(folder), isOn: Binding(get: { draft.pretrainingFolderIDs.contains(folder.id) }, set: {
                        if $0 { draft.pretrainingFolderIDs.insert(folder.id) } else { draft.pretrainingFolderIDs.remove(folder.id) }
                    }))
                }
            }
            Section {
                if hasUnsavedChanges {
                    Text("Unsaved changes · kept while this app is open").font(.caption).foregroundStyle(.orange)
                }
                HStack {
                    Button("Save model configuration") {
                        store.perform {
                            try store.saveModelConfiguration(draft)
                            store.modelDrafts.removeValue(forKey: original.id)
                            store.notice = "Model configuration saved."
                        }
                    }.buttonStyle(.borderedProminent)
                    Button("Discard changes") {
                        draft = saved
                        store.modelDrafts.removeValue(forKey: original.id)
                    }.disabled(!hasUnsavedChanges)
                }
            }
        }.formStyle(.grouped).disabled(!store.activeOperations.isEmpty || store.migrating)
        .onDisappear {
            if hasUnsavedChanges, store.models.contains(where: { $0.id == original.id }) { store.modelDrafts[original.id] = draft }
            else { store.modelDrafts.removeValue(forKey: original.id) }
        }
    }
}

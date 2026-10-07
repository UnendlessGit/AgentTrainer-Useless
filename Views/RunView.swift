import SwiftUI

struct RunView: View {
    var session: AppSession
    @State private var showKeys = false
    private var runner: RunCoordinator { session.runner }
    private var model: AIModel? { session.store.models.first { $0.id == runner.configuration.modelID } }

    var body: some View {
        @Bindable var runner = runner
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(title: "Run your agent", subtitle: "Put what your model has learned into practice.") {
                    if runner.isBusy {
                        Button { runner.stop() } label: { Label("Stop & release input", systemImage: "stop.fill") }
                            .buttonStyle(.borderedProminent).tint(.red)
                    } else {
                        Button { Task { await runner.start() } } label: { Label("Start run", systemImage: "play.fill") }
                            .buttonStyle(.borderedProminent)
                            .disabled(model?.canRun != true || !session.store.activeOperations.isEmpty || session.store.migrating || runner.catalog.loading)
                    }
                }
                HStack(alignment: .top, spacing: 22) {
                    VStack(spacing: 20) {
                        sourcePicker.disabled(runner.isBusy || runner.catalog.loading)
                        ZStack {
                            RoundedRectangle(cornerRadius: 18).fill(.black.opacity(0.92))
                            if let preview = runner.preview { Image(nsImage: preview).resizable().scaledToFit().padding(6) }
                            else {
                                VStack(spacing: 14) {
                                    Image(systemName: "play.circle").font(.system(size: 44, weight: .ultraLight))
                                    Text("Your agent's view").font(.headline)
                                    Text("A three-second countdown gives you time to focus the target.").font(.caption).foregroundStyle(.white.opacity(0.5))
                                }.foregroundStyle(.white.opacity(0.8)).padding()
                            }
                        }.aspectRatio(16 / 10, contentMode: .fit).clipShape(RoundedRectangle(cornerRadius: 18))
                        Surface {
                            HStack {
                                StatusPill(title: runner.phase.rawValue, color: runner.isBusy ? .green : .secondary)
                                Metric(title: "DECISIONS", value: runner.progress.decisions.formatted())
                                Metric(title: "ELAPSED", value: DisplayFormat.duration(runner.progress.elapsed))
                                Metric(title: "INFERENCE", value: String(format: "%.1f ms", runner.progress.inferenceMilliseconds))
                            }
                            Text("\(runner.progress.inputTransitions) input transitions · \(runner.progress.decisions - runner.progress.inputTransitions) waits")
                                .font(.caption).foregroundStyle(.secondary)
                            Text("Held input: " + heldInputLabel)
                                .font(.callout.weight(.medium))
                                .foregroundStyle(heldInputLabel == "None" ? Color.secondary : Color.orange)
                            if !runner.progress.keyPresses.isEmpty {
                                Text("Key presses: " + runner.progress.keyPresses.keys.sorted().map {
                                    "\(KeyNames.name($0)) ×\(runner.progress.keyPresses[$0, default: 0])"
                                }.joined(separator: ", "))
                                .font(.caption).foregroundStyle(.secondary)
                            }
                            if !runner.progress.keyRepeats.isEmpty {
                                Text("Repeated keys: " + runner.progress.keyRepeats.keys.sorted().map {
                                    "\(KeyNames.name($0)) ×\(runner.progress.keyRepeats[$0, default: 0])"
                                }.joined(separator: ", "))
                                .font(.caption).foregroundStyle(.secondary)
                            }
                            Text(runner.message).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                            if runner.progress.waitingForHuman && runner.isBusy {
                                Text("Waiting for you to release keyboard and mouse buttons.").font(.callout).foregroundStyle(.orange)
                            }
                            Text("Emergency stop: \((session.store.preferences.shortcuts ?? ShortcutBindings()).emergency.label)")
                                .font(.callout.weight(.medium))
                        }
                        if !runner.progress.history.isEmpty {
                            Surface(title: "Recent actions", symbol: "list.bullet") {
                                ForEach(Array(runner.progress.history.suffix(12).enumerated()), id: \.offset) { _, action in
                                    Text(action).font(.system(.caption, design: .monospaced)).frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                        }
                    }.frame(maxWidth: .infinity)
                    VStack(spacing: 20) {
                        Surface(title: "Agent", symbol: "cpu") {
                            Picker("Model", selection: $runner.configuration.modelID) {
                                Text("Select a trained model").tag(nil as UUID?)
                                ForEach(session.store.models.filter(\.canRun)) { Text($0.name).tag(Optional($0.id)) }
                            }
                            TextField("Task instruction", text: $runner.configuration.instruction, axis: .vertical).lineLimit(2...4)
                            Toggle("Use best validation checkpoint", isOn: $runner.configuration.useBestCheckpoint)
                            Text("Uses the latest checkpoint if no best checkpoint matches this configuration and dataset.").font(.caption).foregroundStyle(.secondary)
                            Toggle("Choose the most likely action", isOn: $runner.configuration.deterministic)
                            if !runner.configuration.deterministic {
                                Slider(value: $runner.configuration.temperature, in: 0.05...2) { Text("Sampling temperature") }
                                Text("Temperature \(runner.configuration.temperature, format: .number.precision(.fractionLength(2)))").font(.caption)
                            }
                        }
                        Surface(title: "Control limits", symbol: "hand.raised") {
                            Toggle("Stop on human input", isOn: $runner.configuration.stopOnHumanInput)
                            if !runner.configuration.stopOnHumanInput {
                                Text("The agent waits while you hold input. Conflicting input stops the run while the agent holds a key or button.").font(.caption).foregroundStyle(.secondary)
                            }
                            Picker("Run duration", selection: $runner.configuration.maximumRunSeconds) {
                                ForEach([15, 30, 60, 120, 300, 600], id: \.self) { Text("\($0) seconds").tag($0) }
                            }
                            Stepper("Maximum hold: \(runner.configuration.maximumHoldSeconds)s", value: $runner.configuration.maximumHoldSeconds, in: 1...60)
                            Toggle("Move pointer", isOn: $runner.configuration.permissions.pointer)
                            Toggle("Relative pointer movement", isOn: $runner.configuration.permissions.relativePointer)
                            Toggle("Scrolling", isOn: $runner.configuration.permissions.scrolling)
                            Toggle("Dragging", isOn: $runner.configuration.permissions.dragging)
                            Toggle("Key combinations", isOn: $runner.configuration.permissions.chords)
                            Toggle("Repeated key events while held", isOn: $runner.configuration.permissions.repeatsKeys)
                            ForEach(0...2, id: \.self) { button in
                                Toggle(KeyNames.button(button), isOn: Binding(get: { runner.configuration.permissions.buttons.contains(button) }, set: {
                                    if $0 { runner.configuration.permissions.buttons.insert(button) } else { runner.configuration.permissions.buttons.remove(button) }
                                }))
                            }
                            Text("Run permissions can restrict a model's trained capabilities.").font(.caption).foregroundStyle(.secondary)
                            Button(showKeys ? "Hide keyboard permissions" : "Choose allowed keys") { showKeys.toggle() }
                            if showKeys {
                                Button("Clear keys") { runner.configuration.permissions.keys.removeAll() }
                                Button("Allow model's keys") { runner.configuration.permissions.keys = model?.configuration.capabilities.keys ?? [] }
                                LazyVGrid(columns: [GridItem(.adaptive(minimum: 105))]) {
                                    ForEach((model?.configuration.capabilities.keys ?? []).sorted(), id: \.self) { key in
                                        KeyPermissionToggle(key: key, isOn: Binding(get: { runner.configuration.permissions.keys.contains(key) }, set: {
                                            if $0 { runner.configuration.permissions.keys.insert(key) } else { runner.configuration.permissions.keys.remove(key) }
                                        }))
                                    }
                                }
                            }
                        }
                    }.frame(width: 320).disabled(runner.isBusy)
                }
            }.padding(30)
        }
        .task {
            if runner.configuration.modelID == nil { runner.configuration.modelID = session.store.models.first(where: \.canRun)?.id }
            if runner.catalog.displays.isEmpty { await refresh() }
        }
        .onChange(of: runner.configuration.modelID) { _, _ in if let model { runner.configuration.permissions = model.configuration.capabilities } }
        .onDisappear { runner.saveConfiguration() }
    }

    private var sourcePicker: some View {
        @Bindable var runner = runner
        return Surface(title: "Where should the agent work?") {
            Picker("Capture target", selection: $runner.configuration.capture.target.kind) {
                ForEach(CaptureKind.allCases) { Label($0.rawValue, systemImage: $0.symbol).tag($0) }
            }.pickerStyle(.segmented).labelsHidden()
            HStack {
                if runner.configuration.capture.target.kind == .desktop { Label("All connected displays", systemImage: "display.2").foregroundStyle(.secondary) }
                else if runner.configuration.capture.target.kind == .window || (runner.configuration.capture.target.kind == .region && runner.configuration.capture.regionWindow) {
                    Picker("Window", selection: $runner.configuration.capture.target.windowID) {
                        Text("Choose a window").tag(nil as UInt32?)
                        ForEach(runner.catalog.windows) { Text($0.title).tag(Optional($0.id)) }
                    }.labelsHidden()
                } else {
                    Picker("Display", selection: $runner.configuration.capture.target.displayID) {
                        Text("Choose a display").tag(nil as UInt32?)
                        ForEach(runner.catalog.displays) { Text($0.title).tag(Optional($0.id)) }
                    }.labelsHidden()
                }
                Spacer(minLength: 4)
                Button { Task { await refresh() } } label: { Image(systemName: "arrow.clockwise") }.help("Refresh capture sources")
            }
            if runner.configuration.capture.target.kind == .region {
                Toggle("Region of a window", isOn: $runner.configuration.capture.regionWindow)
                HStack {
                    TextField("X", value: $runner.configuration.capture.regionX, format: .number)
                    TextField("Y", value: $runner.configuration.capture.regionY, format: .number)
                    TextField("Width", value: $runner.configuration.capture.regionWidth, format: .number)
                    TextField("Height", value: $runner.configuration.capture.regionHeight, format: .number)
                }
                Text("Coordinates in points from the source's top-left corner.").font(.caption).foregroundStyle(.secondary)
            }
            if let error = runner.catalog.error { Text(error).font(.caption).foregroundStyle(.red) }
        }
    }

    private func refresh() async {
        await runner.catalog.refresh()
        if !runner.catalog.displays.contains(where: { $0.id == runner.configuration.capture.target.displayID }) { runner.configuration.capture.target.displayID = runner.catalog.displays.first?.id }
        if let id = runner.configuration.capture.target.windowID, !runner.catalog.windows.contains(where: { $0.id == id }) { runner.configuration.capture.target.windowID = nil }
    }

    private var heldInputLabel: String {
        let state = runner.progress.heldInput
        let labels = state.keys.sorted().map(KeyNames.name) + state.buttons.sorted().map(KeyNames.button)
        return labels.isEmpty ? "None" : labels.joined(separator: ", ")
    }
}

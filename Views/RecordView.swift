import SwiftUI

struct RecordView: View {
    var session: AppSession
    @State private var name = "Untitled demonstration"
    @State private var instruction = ""
    @State private var folderID: UUID?
    @State private var target = CaptureTarget()
    @State private var settings = RecordingSettings()
    @State private var regionX = 0.0
    @State private var regionY = 0.0
    @State private var regionWidth = 800.0
    @State private var regionHeight = 600.0
    @State private var regionWindow = false
    private var recorder: RecordingCoordinator { session.recorder }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(title: "Record a demonstration", subtitle: "Show your model how you work. Every action, in time.") {
                    if recorder.isBusy {
                        Button { Task { await recorder.stop() } } label: { Label("Stop recording", systemImage: "stop.fill") }
                            .buttonStyle(.borderedProminent).tint(.red).disabled(recorder.phase == .stopping)
                    } else {
                        Button(action: start) { Label("Start recording", systemImage: "record.circle") }
                            .buttonStyle(.borderedProminent).controlSize(.large)
                            .accessibilityLabel("Start recording").accessibilityIdentifier("record.start")
                    }
                }
                HStack(alignment: .top, spacing: 22) {
                    VStack(spacing: 20) {
                        captureSelection.disabled(recorder.isBusy)
                        preview
                        Surface {
                            HStack {
                                Metric(title: "DURATION", value: DisplayFormat.duration(recorder.manifest?.duration ?? 0))
                                Metric(title: "OBSERVATIONS", value: (recorder.manifest?.observationCount ?? 0).formatted())
                                Metric(title: "INPUT EVENTS", value: (recorder.manifest?.inputEventCount ?? 0).formatted())
                                StatusPill(title: recorder.phase.rawValue, color: recorder.isBusy ? .red : .secondary)
                            }
                        }
                        if !recorder.permissions.screenRecording {
                            Surface(title: "Allow screen recording", symbol: "lock.rectangle") {
                                Text("AgentTrainer needs permission to capture your selected screen or window. You control what is recorded.").foregroundStyle(.secondary)
                                Button("Open permissions") { session.tab = .settings }
                            }
                        }
                    }.frame(maxWidth: .infinity)
                    VStack(spacing: 20) {
                        Surface(title: "Save to Library", symbol: "folder") {
                            VStack(alignment: .leading, spacing: 7) { Text("Recording name").font(.caption).foregroundStyle(.secondary); TextField("Recording name", text: $name).accessibilityIdentifier("record.name") }
                            Picker("Folder", selection: $folderID) {
                                Text("Choose folder").tag(nil as UUID?)
                                ForEach(session.store.folders) { folder in
                                    Text("\(folder.kind.rawValue) / \(session.store.folderPath(folder))").tag(Optional(folder.id))
                                }
                            }
                            VStack(alignment: .leading, spacing: 7) {
                                Text("Task instruction · optional").font(.caption).foregroundStyle(.secondary)
                                TextField("What are you demonstrating?", text: $instruction, axis: .vertical).lineLimit(2...4)
                            }
                        }
                        Surface(title: "Capture quality", symbol: "viewfinder") {
                            Picker("Resolution", selection: $settings.maximumDimension) {
                                Text("720 px").tag(720); Text("1280 px").tag(1280); Text("1920 px").tag(1920); Text("2560 px").tag(2560)
                            }
                            Picker("Capture rate", selection: $settings.framesPerSecond) {
                                ForEach([5, 10, 15, 30, 60], id: \.self) { Text("\($0) observations/s").tag($0) }
                            }
                            VStack(alignment: .leading, spacing: 6) {
                                HStack { Text("Image quality"); Spacer(); Text(settings.quality, format: .percent.precision(.fractionLength(0))).foregroundStyle(.secondary) }
                                Slider(value: $settings.quality, in: 0.5...1, step: 0.05)
                            }
                            Toggle("Include cursor", isOn: $settings.includesCursor)
                        }
                        Surface(title: "Input capture", symbol: "keyboard") {
                            Toggle("Keyboard transitions", isOn: $settings.keyboard)
                            Toggle("Mouse buttons", isOn: $settings.mouseButtons)
                            Toggle("Pointer movement", isOn: $settings.pointerMovement).disabled(settings.relativeMovement)
                            Toggle("Scrolling", isOn: $settings.scrolling)
                            Toggle("Relative mouse movement", isOn: $settings.relativeMovement)
                            Text("Input transitions are captured independently of the visual capture rate.")
                                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                    }.frame(width: 310).disabled(recorder.isBusy)
                }
            }.padding(30)
        }
        .task {
            recorder.permissions.refresh()
            if recorder.permissions.screenRecording && recorder.catalog.displays.isEmpty { await refresh() }
            if folderID == nil { folderID = session.store.folders.first(where: { $0.kind == .imitation })?.id }
        }
        .onChange(of: session.store.folders) { _, folders in
            if !folders.contains(where: { $0.id == folderID }) { folderID = folders.first(where: { $0.kind == .imitation })?.id }
        }
    }

    private var captureSelection: some View {
        Surface(title: "What would you like to capture?") {
            Picker("Capture target", selection: $target.kind) {
                ForEach(CaptureKind.allCases) { Label($0.rawValue, systemImage: $0.symbol).tag($0) }
            }.pickerStyle(.segmented).labelsHidden()
            HStack {
                if target.kind == .desktop { Label("All connected displays", systemImage: "display.2").foregroundStyle(.secondary) }
                else if target.kind == .window || (target.kind == .region && regionWindow) {
                    Picker("Window", selection: $target.windowID) {
                        Text("Choose a window").tag(nil as UInt32?)
                        ForEach(recorder.catalog.windows) { Text($0.title).tag(Optional($0.id)) }
                    }.labelsHidden()
                } else {
                    Picker("Display", selection: $target.displayID) {
                        Text("Choose a display").tag(nil as UInt32?)
                        ForEach(recorder.catalog.displays) { Text($0.title).tag(Optional($0.id)) }
                    }.labelsHidden()
                }
                Spacer(minLength: 4)
                Button { Task { await refresh() } } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh displays and windows").disabled(recorder.catalog.loading)
            }
            if target.kind == .region {
                Toggle("Region of a window", isOn: $regionWindow)
                HStack {
                    regionField("X", value: $regionX); regionField("Y", value: $regionY)
                    regionField("Width", value: $regionWidth); regionField("Height", value: $regionHeight)
                }
                Text("Coordinates in points, measured from the top-left of the selected source.").font(.caption).foregroundStyle(.secondary)
            }
            if let error = recorder.catalog.error { Text(error).font(.caption).foregroundStyle(.red) }
        }
    }

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(.black.opacity(0.92))
            if let image = recorder.preview {
                Image(nsImage: image).resizable().scaledToFit().padding(6)
            } else {
                VStack(spacing: 15) {
                    Image(systemName: "viewfinder").font(.system(size: 40, weight: .ultraLight))
                    Text("Your demonstration starts here").font(.headline)
                    Text("Choose a source, then start recording.").font(.callout).foregroundStyle(.white.opacity(0.45))
                }.foregroundStyle(.white.opacity(0.8))
            }
        }.aspectRatio(16 / 10, contentMode: .fit).clipShape(RoundedRectangle(cornerRadius: 18))
    }

    private func regionField(_ title: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title).font(.caption).foregroundStyle(.secondary); TextField(title, value: value, format: .number.precision(.fractionLength(0))) }
    }
    private func refresh() async {
        await recorder.catalog.refresh()
        if target.displayID == nil { target.displayID = recorder.catalog.displays.first?.id }
    }
    private func start() {
        var resolved = target
        if target.kind == .region {
            resolved.region = CaptureRect(CGRect(x: regionX, y: regionY, width: regionWidth, height: regionHeight))
            if !regionWindow { resolved.windowID = nil }
        }
        if target.kind == .display || target.kind == .desktop { resolved.windowID = nil }
        resolved.title = target.kind == .window || (target.kind == .region && regionWindow)
            ? recorder.catalog.windows.first(where: { $0.id == target.windowID })?.title ?? "Window"
            : target.kind == .desktop ? "Full desktop" : recorder.catalog.displays.first(where: { $0.id == target.displayID })?.title ?? "Display"
        Task { await recorder.start(name: name, instruction: instruction, folderID: folderID, target: resolved, settings: settings) }
    }
}

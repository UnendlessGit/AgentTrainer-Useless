import SwiftUI

struct RunView: View {
    var session: AppSession
    @State private var modelID: UUID?
    @State private var instruction = ""
    @State private var stopOnHuman = true
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(title: "Run your agent", subtitle: "Let your model put what it has learned into practice.") { StatusPill(title: "Stopped") }
                Surface(title: "Agent", symbol: "cpu") {
                    Picker("Trained model", selection: $modelID) {
                        Text("Select a trained model").tag(nil as UUID?)
                        ForEach(session.store.models.filter(\.canRun)) { Text($0.name).tag(Optional($0.id)) }
                    }
                    TextField("Task instruction", text: $instruction)
                    Toggle("Stop when I use the keyboard or mouse", isOn: $stopOnHuman)
                }
                EmptyState(symbol: "play.circle", title: "A trained model is the next step", message: "Run becomes available after a compatible imitation-learning checkpoint has been created and the execution pipeline is connected.")
                    .frame(minHeight: 330)
            }.padding(30)
        }
    }
}

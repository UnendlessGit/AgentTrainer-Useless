import SwiftUI

struct InstructionSizeHint: View {
    var text: String
    private var count: Int { text.utf8.count }
    private var limit: Int { PolicyNetwork.instructionLength }
    var body: some View {
        Text(count > limit ? "\(count)/\(limit) UTF-8 bytes · shorten before training or running with task instructions"
             : "\(count)/\(limit) UTF-8 bytes")
            .font(.caption).foregroundStyle(count > limit ? Color.orange : Color.secondary)
            .help("Spaces and punctuation count. Some characters use multiple bytes. Recording metadata preserves the complete text.")
            .fixedSize(horizontal: false, vertical: true)
    }
}

struct PageHeader<Actions: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder var actions: Actions
    var body: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 8) {
                Text(title).font(.system(size: 29, weight: .semibold, design: .rounded))
                Text(subtitle).font(.callout).foregroundStyle(.secondary)
            }
            Spacer(minLength: 20)
            actions
        }.padding(.bottom, 14)
    }
}

struct Surface<Content: View>: View {
    var title: String? = nil
    var symbol: String? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let title {
                HStack(spacing: 8) {
                    if let symbol { Image(systemName: symbol).foregroundStyle(.secondary) }
                    Text(title).font(.headline)
                }
            }
            content
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(.primary.opacity(0.07), lineWidth: 1))
    }
}

struct Metric: View {
    let title: String
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded, weight: .medium)).monospacedDigit()
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct EmptyState: View {
    let symbol: String
    let title: String
    let message: String
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: symbol).font(.system(size: 34, weight: .light)).foregroundStyle(.secondary)
            Text(title).font(.title3.weight(.medium))
            Text(message).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 380)
        }.padding(40).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct StatusPill: View {
    var title: String
    var color: Color = .secondary
    var body: some View {
        HStack(spacing: 6) { Circle().fill(color).frame(width: 6, height: 6); Text(title).font(.caption.weight(.medium)) }
            .padding(.horizontal, 10).padding(.vertical, 6).background(color.opacity(0.08), in: Capsule())
    }
}

enum DisplayFormat {
    static func duration(_ seconds: Double) -> String {
        let total = Int(max(0, seconds))
        return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, (total % 3600) / 60, total % 60)
            : String(format: "%02d:%02d", total / 60, total % 60)
    }
}

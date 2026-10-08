import AppKit

// Draw every size from the same vector geometry; no external artwork or font
// dependency. The capture corners and learning bars match the workspace accent.
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let sizes = [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2)]
var entries: [[String: String]] = []
for (size, scale) in sizes {
    let pixels = size * scale
    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
        let context = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("Cannot create icon bitmap") }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.cgContext.scaleBy(x: CGFloat(pixels) / 1024, y: CGFloat(pixels) / 1024)
    let background = NSBezierPath(roundedRect: NSRect(x: 72, y: 72, width: 880, height: 880), xRadius: 198, yRadius: 198)
    NSGradient(starting: NSColor(srgbRed: 0.06, green: 0.66, blue: 0.99, alpha: 1),
               ending: NSColor(srgbRed: 0.02, green: 0.30, blue: 0.91, alpha: 1))!.draw(in: background, angle: -70)
    NSColor.white.setStroke()
    for (x, y, dx, dy) in [(270.0, 270.0, 1.0, 1.0), (754.0, 270.0, -1.0, 1.0),
                           (270.0, 754.0, 1.0, -1.0), (754.0, 754.0, -1.0, -1.0)] {
        let corner = NSBezierPath()
        corner.move(to: NSPoint(x: x, y: y + dy * 100))
        corner.line(to: NSPoint(x: x, y: y + dy * 22))
        corner.curve(to: NSPoint(x: x + dx * 22, y: y),
                     controlPoint1: NSPoint(x: x, y: y + dy * 7), controlPoint2: NSPoint(x: x + dx * 7, y: y))
        corner.line(to: NSPoint(x: x + dx * 100, y: y))
        corner.lineWidth = 48; corner.lineCapStyle = .round; corner.lineJoinStyle = .round
        corner.stroke()
    }
    NSColor.white.setFill()
    for (x, height) in [(378.0, 112.0), (484.0, 190.0), (590.0, 276.0)] {
        NSBezierPath(roundedRect: NSRect(x: x, y: 374, width: 58, height: height), xRadius: 22, yRadius: 22).fill()
    }
    NSGraphicsContext.restoreGraphicsState()
    let filename = "icon_\(size)x\(size)@\(scale)x.png"
    try bitmap.representation(using: .png, properties: [:])!.write(to: output.appendingPathComponent(filename))
    entries.append(["idiom": "mac", "size": "\(size)x\(size)", "scale": "\(scale)x", "filename": filename])
}
let catalog: [String: Any] = ["images": entries, "info": ["author": "xcode", "version": 1]]
try JSONSerialization.data(withJSONObject: catalog, options: [.prettyPrinted, .sortedKeys])
    .write(to: output.appendingPathComponent("Contents.json"))

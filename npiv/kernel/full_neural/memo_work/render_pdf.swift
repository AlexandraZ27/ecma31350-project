import Foundation
import PDFKit
import AppKit

let args = CommandLine.arguments
let pdf = PDFDocument(url: URL(fileURLWithPath: args[1]))!
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let pages = args.count > 3 ? args[3].split(separator: ",").compactMap { Int($0) } : Array(1...pdf.pageCount)
for number in pages {
    let page = pdf.page(at: number - 1)!
    let bounds = page.bounds(for: .mediaBox)
    let image = page.thumbnail(of: NSSize(width: 1200, height: 1200 * bounds.height / bounds.width), for: .mediaBox)
    let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
    try bitmap.representation(using: .png, properties: [:])!.write(to: output.appendingPathComponent("page-\(number).png"))
    try (page.string ?? "").write(to: output.appendingPathComponent("page-\(number).txt"), atomically: true, encoding: .utf8)
}
print("Rendered \(pages.count) pages from \(pdf.pageCount)-page PDF")

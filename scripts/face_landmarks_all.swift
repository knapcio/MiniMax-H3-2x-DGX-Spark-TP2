import Foundation
import Vision
import ImageIO
import CoreGraphics

func detect(_ path: String) throws -> [String: Any] {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "all-face-probe", code: 1, userInfo: [NSLocalizedDescriptionKey: "image load failed"])
    }
    let request = VNDetectFaceLandmarksRequest()
    request.usesCPUOnly = true
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    var output: [[String: Any]] = []
    for face in request.results ?? [] where face.confidence >= 0.5 {
        var row: [String: Any] = ["confidence": face.confidence,
            "face_box": [face.boundingBox.minX, face.boundingBox.minY, face.boundingBox.width, face.boundingBox.height]]
        if let lips = face.landmarks?.innerLips, lips.pointCount > 2 {
            let points = lips.normalizedPoints
            let xs = (0..<lips.pointCount).map { Double(points[$0].x) }
            let ys = (0..<lips.pointCount).map { Double(points[$0].y) }
            let width = (xs.max() ?? 0) - (xs.min() ?? 0)
            let height = (ys.max() ?? 0) - (ys.min() ?? 0)
            if width > 0 { row["inner_lip_height_over_width"] = height / width }
        }
        output.append(row)
    }
    return ["path": path, "faces": output]
}
let paths = Array(CommandLine.arguments.dropFirst())
guard !paths.isEmpty else { fputs("usage: face_landmarks_all FRAME...\n", stderr); exit(2) }
var records: [[String: Any]] = []
for path in paths {
    do { records.append(try detect(path)) }
    catch { records.append(["path":path, "faces":[], "error":error.localizedDescription]) }
}
do { let data = try JSONSerialization.data(withJSONObject: records, options: [.sortedKeys]); print(String(decoding:data, as:UTF8.self)) }
catch { fputs("json failure\n", stderr); exit(1) }

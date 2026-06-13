import CoreGraphics
import CoreML
import Foundation

// Reference decoder for the Wheels CoreML handoff model:
// input image: 384x384
// output: var_1344, logical shape [1, 14, 3024]
//
// Channel layout:
// 0..3  = bbox center-x, center-y, width, height in 384x384 model pixels
// 4     = wheel confidence
// 5..13 = three keypoints: a(x,y,score), b(x,y,score), c_disc_bottom(x,y,score)

public enum WheelsResizeMode {
    // Use when the app manually resized the original frame directly to 384x384.
    // Reverse mapping uses independent x/y scales.
    case stretch

    // Use when the frame was aspect-fit into 384x384 with padding.
    case scaleFitLetterbox

    // Use when the frame was aspect-filled into 384x384 and center-cropped.
    case centerCrop
}

public struct WheelsOutputStats: Equatable {
    public let count: Int
    public let finiteCount: Int
    public let nonFiniteCount: Int
    public let min: Double
    public let max: Double
    public let mean: Double
    public let allZero: Bool
}

public struct WheelsPoints: Codable, Equatable {
    public let a: [Double]
    public let b: [Double]
    public let cDiscBottom: [Double]

    enum CodingKeys: String, CodingKey {
        case a
        case b
        case cDiscBottom = "c_disc_bottom"
    }
}

public struct WheelDetection: Codable, Equatable {
    public let bboxXYXY: [Double]
    public let confidence: Double
    public let points: WheelsPoints

    enum CodingKeys: String, CodingKey {
        case bboxXYXY = "bbox_xyxy"
        case confidence
        case points
    }
}

public struct WheelsPayload: Codable, Equatable {
    public let frameID: String
    public let wheels: [WheelDetection]

    enum CodingKeys: String, CodingKey {
        case frameID = "frame_id"
        case wheels
    }
}

public enum WheelsPoseDecoderError: Error, CustomStringConvertible {
    case invalidOriginalSize(CGSize)
    case invalidThresholds(confidence: Double, iou: Double, maxDetections: Int)
    case unexpectedOutputShape(shape: [Int], count: Int)

    public var description: String {
        switch self {
        case .invalidOriginalSize(let size):
            return "Invalid original frame size: \(size.width)x\(size.height)"
        case .invalidThresholds(let confidence, let iou, let maxDetections):
            return "Invalid thresholds: confidence=\(confidence), iou=\(iou), maxDetections=\(maxDetections)"
        case .unexpectedOutputShape(let shape, let count):
            return "Expected CoreML output var_1344 with logical shape [1, 14, 3024], got shape=\(shape), count=\(count)"
        }
    }
}

public enum WheelsPoseDecoder {
    public static let inputSize: Double = 384.0
    public static let channelCount: Int = 14
    public static let anchorCount: Int = 3024
    public static let expectedValueCount: Int = channelCount * anchorCount

    public static func outputStats(_ output: MLMultiArray) -> WheelsOutputStats {
        var minValue = Double.infinity
        var maxValue = -Double.infinity
        var sum = 0.0
        var finiteCount = 0
        var nonFiniteCount = 0
        var allZero = true

        for index in 0..<output.count {
            let value = output[index].doubleValue
            if value.isFinite {
                finiteCount += 1
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
                sum += value
                if value != 0.0 {
                    allZero = false
                }
            } else {
                nonFiniteCount += 1
            }
        }

        if finiteCount == 0 {
            minValue = .nan
            maxValue = .nan
        }

        return WheelsOutputStats(
            count: output.count,
            finiteCount: finiteCount,
            nonFiniteCount: nonFiniteCount,
            min: minValue,
            max: maxValue,
            mean: finiteCount > 0 ? sum / Double(finiteCount) : .nan,
            allZero: allZero && nonFiniteCount == 0
        )
    }

    public static func decode(
        output: MLMultiArray,
        originalSize: CGSize,
        resizeMode: WheelsResizeMode = .stretch,
        confidenceThreshold: Double = 0.25,
        iouThreshold: Double = 0.45,
        maxDetections: Int = 20
    ) throws -> [WheelDetection] {
        guard originalSize.width > 0, originalSize.height > 0 else {
            throw WheelsPoseDecoderError.invalidOriginalSize(originalSize)
        }
        guard confidenceThreshold >= 0.0,
              confidenceThreshold <= 1.0,
              iouThreshold >= 0.0,
              iouThreshold <= 1.0,
              maxDetections > 0 else {
            throw WheelsPoseDecoderError.invalidThresholds(
                confidence: confidenceThreshold,
                iou: iouThreshold,
                maxDetections: maxDetections
            )
        }

        let view = try TensorView(output)
        var candidates: [Candidate] = []
        candidates.reserveCapacity(anchorCount)

        for anchor in 0..<anchorCount {
            let score = view.value(channel: 4, anchor: anchor)
            guard score.isFinite, score >= confidenceThreshold else {
                continue
            }

            let cx = view.value(channel: 0, anchor: anchor)
            let cy = view.value(channel: 1, anchor: anchor)
            let width = view.value(channel: 2, anchor: anchor)
            let height = view.value(channel: 3, anchor: anchor)
            guard cx.isFinite, cy.isFinite, width.isFinite, height.isFinite,
                  width > 1.0, height > 1.0 else {
                continue
            }

            let bbox = BBox(
                x1: clamp(cx - width / 2.0, min: 0.0, max: inputSize),
                y1: clamp(cy - height / 2.0, min: 0.0, max: inputSize),
                x2: clamp(cx + width / 2.0, min: 0.0, max: inputSize),
                y2: clamp(cy + height / 2.0, min: 0.0, max: inputSize)
            )
            guard bbox.width > 1.0, bbox.height > 1.0 else {
                continue
            }

            var keypoints: [Keypoint] = []
            keypoints.reserveCapacity(3)
            for keypointIndex in 0..<3 {
                let base = 5 + keypointIndex * 3
                let x = view.value(channel: base, anchor: anchor)
                let y = view.value(channel: base + 1, anchor: anchor)
                let keypointScore = view.value(channel: base + 2, anchor: anchor)
                guard x.isFinite, y.isFinite else {
                    keypoints.append(Keypoint(x: 0.0, y: 0.0, score: 0.0))
                    continue
                }
                keypoints.append(
                    Keypoint(
                        x: clamp(x, min: 0.0, max: inputSize),
                        y: clamp(y, min: 0.0, max: inputSize),
                        score: keypointScore.isFinite ? keypointScore : 0.0
                    )
                )
            }

            candidates.append(
                Candidate(
                    bbox: bbox,
                    confidence: score,
                    keypoints: keypoints
                )
            )
        }

        let kept = nonMaxSuppression(
            candidates.sorted { $0.confidence > $1.confidence },
            iouThreshold: iouThreshold,
            maxDetections: maxDetections
        )

        return kept.map {
            detection(from: $0, originalSize: originalSize, resizeMode: resizeMode)
        }
    }

    public static func decodePayload(
        output: MLMultiArray,
        frameID: String,
        originalSize: CGSize,
        resizeMode: WheelsResizeMode = .stretch,
        confidenceThreshold: Double = 0.25,
        iouThreshold: Double = 0.45,
        maxDetections: Int = 20
    ) throws -> WheelsPayload {
        let wheels = try decode(
            output: output,
            originalSize: originalSize,
            resizeMode: resizeMode,
            confidenceThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            maxDetections: maxDetections
        )
        return WheelsPayload(frameID: frameID, wheels: wheels)
    }

    public static func encodePayloadJSON(_ payload: WheelsPayload) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(payload)
    }
}

private struct TensorView {
    private let output: MLMultiArray
    private let shape: [Int]
    private let channelDimension: Int?
    private let anchorDimension: Int?

    init(_ output: MLMultiArray) throws {
        self.output = output
        self.shape = output.shape.map { $0.intValue }

        guard output.count == WheelsPoseDecoder.expectedValueCount else {
            throw WheelsPoseDecoderError.unexpectedOutputShape(
                shape: shape,
                count: output.count
            )
        }

        if shape.count == 1 {
            self.channelDimension = nil
            self.anchorDimension = nil
            return
        }

        guard let channelDimension = shape.firstIndex(of: WheelsPoseDecoder.channelCount),
              let anchorDimension = shape.firstIndex(of: WheelsPoseDecoder.anchorCount) else {
            throw WheelsPoseDecoderError.unexpectedOutputShape(
                shape: shape,
                count: output.count
            )
        }

        for (dimensionIndex, dimensionSize) in shape.enumerated() {
            if dimensionIndex != channelDimension,
               dimensionIndex != anchorDimension,
               dimensionSize != 1 {
                throw WheelsPoseDecoderError.unexpectedOutputShape(
                    shape: shape,
                    count: output.count
                )
            }
        }

        self.channelDimension = channelDimension
        self.anchorDimension = anchorDimension
    }

    func value(channel: Int, anchor: Int) -> Double {
        if shape.count == 1 {
            return output[channel * WheelsPoseDecoder.anchorCount + anchor].doubleValue
        }

        var indices = Array(repeating: NSNumber(value: 0), count: shape.count)
        indices[channelDimension!] = NSNumber(value: channel)
        indices[anchorDimension!] = NSNumber(value: anchor)
        return output[indices].doubleValue
    }
}

private struct Candidate {
    let bbox: BBox
    let confidence: Double
    let keypoints: [Keypoint]
}

private struct Keypoint {
    let x: Double
    let y: Double
    let score: Double
}

private struct Point {
    let x: Double
    let y: Double
}

private struct BBox {
    let x1: Double
    let y1: Double
    let x2: Double
    let y2: Double

    var width: Double { x2 - x1 }
    var height: Double { y2 - y1 }
    var area: Double { max(0.0, width) * max(0.0, height) }
}

private func detection(
    from candidate: Candidate,
    originalSize: CGSize,
    resizeMode: WheelsResizeMode
) -> WheelDetection {
    let p1 = mapPoint(
        Point(x: candidate.bbox.x1, y: candidate.bbox.y1),
        originalSize: originalSize,
        resizeMode: resizeMode
    )
    let p2 = mapPoint(
        Point(x: candidate.bbox.x2, y: candidate.bbox.y2),
        originalSize: originalSize,
        resizeMode: resizeMode
    )

    let x1 = clamp(min(p1.x, p2.x), min: 0.0, max: Double(originalSize.width))
    let y1 = clamp(min(p1.y, p2.y), min: 0.0, max: Double(originalSize.height))
    let x2 = clamp(max(p1.x, p2.x), min: 0.0, max: Double(originalSize.width))
    let y2 = clamp(max(p1.y, p2.y), min: 0.0, max: Double(originalSize.height))

    let mapped = candidate.keypoints.map {
        mapPoint(Point(x: $0.x, y: $0.y), originalSize: originalSize, resizeMode: resizeMode)
    }

    return WheelDetection(
        bboxXYXY: [x1, y1, x2, y2],
        confidence: candidate.confidence,
        points: WheelsPoints(
            a: pointArray(mapped[safe: 0] ?? Point(x: 0.0, y: 0.0), originalSize: originalSize),
            b: pointArray(mapped[safe: 1] ?? Point(x: 0.0, y: 0.0), originalSize: originalSize),
            cDiscBottom: pointArray(mapped[safe: 2] ?? Point(x: 0.0, y: 0.0), originalSize: originalSize)
        )
    )
}

private func pointArray(_ point: Point, originalSize: CGSize) -> [Double] {
    [
        clamp(point.x, min: 0.0, max: Double(originalSize.width)),
        clamp(point.y, min: 0.0, max: Double(originalSize.height))
    ]
}

private func mapPoint(
    _ point: Point,
    originalSize: CGSize,
    resizeMode: WheelsResizeMode
) -> Point {
    let originalWidth = Double(originalSize.width)
    let originalHeight = Double(originalSize.height)

    switch resizeMode {
    case .stretch:
        return Point(
            x: point.x * originalWidth / WheelsPoseDecoder.inputSize,
            y: point.y * originalHeight / WheelsPoseDecoder.inputSize
        )

    case .scaleFitLetterbox:
        let scale = min(
            WheelsPoseDecoder.inputSize / originalWidth,
            WheelsPoseDecoder.inputSize / originalHeight
        )
        let padX = (WheelsPoseDecoder.inputSize - originalWidth * scale) / 2.0
        let padY = (WheelsPoseDecoder.inputSize - originalHeight * scale) / 2.0
        return Point(x: (point.x - padX) / scale, y: (point.y - padY) / scale)

    case .centerCrop:
        let scale = max(
            WheelsPoseDecoder.inputSize / originalWidth,
            WheelsPoseDecoder.inputSize / originalHeight
        )
        let offsetX = (WheelsPoseDecoder.inputSize - originalWidth * scale) / 2.0
        let offsetY = (WheelsPoseDecoder.inputSize - originalHeight * scale) / 2.0
        return Point(x: (point.x - offsetX) / scale, y: (point.y - offsetY) / scale)
    }
}

private func nonMaxSuppression(
    _ candidates: [Candidate],
    iouThreshold: Double,
    maxDetections: Int
) -> [Candidate] {
    var kept: [Candidate] = []

    for candidate in candidates {
        if kept.count >= maxDetections {
            break
        }

        let overlapsKept = kept.contains {
            iou(candidate.bbox, $0.bbox) > iouThreshold
        }

        if !overlapsKept {
            kept.append(candidate)
        }
    }

    return kept
}

private func iou(_ a: BBox, _ b: BBox) -> Double {
    let x1 = max(a.x1, b.x1)
    let y1 = max(a.y1, b.y1)
    let x2 = min(a.x2, b.x2)
    let y2 = min(a.y2, b.y2)
    let intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    let union = a.area + b.area - intersection
    guard union > 0.0 else {
        return 0.0
    }
    return intersection / union
}

private func clamp(_ value: Double, min lower: Double, max upper: Double) -> Double {
    Swift.max(lower, Swift.min(upper, value))
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

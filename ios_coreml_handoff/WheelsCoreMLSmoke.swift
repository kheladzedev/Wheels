import CoreML
import CoreVideo
import Foundation

enum WheelsCoreMLSmokeError: Error, CustomStringConvertible {
    case usage
    case pixelBufferCreation
    case missingOutput(String)
    case badOutputCount(Int)

    var description: String {
        switch self {
        case .usage:
            return "Usage: WheelsCoreMLSmoke /path/to/best.mlmodelc"
        case .pixelBufferCreation:
            return "Failed to create 640x640 pixel buffer"
        case let .missingOutput(name):
            return "Missing CoreML output: \(name)"
        case let .badOutputCount(count):
            return "Unexpected output value count: \(count), expected 117600"
        }
    }
}

func makeZeroPixelBuffer(width: Int = 640, height: Int = 640) throws -> CVPixelBuffer {
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        kCVPixelBufferMetalCompatibilityKey as String: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attrs as CFDictionary,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer else {
        throw WheelsCoreMLSmokeError.pixelBufferCreation
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
        let byteCount = CVPixelBufferGetDataSize(pixelBuffer)
        memset(base, 0, byteCount)
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
    return pixelBuffer
}

func runSmoke(modelURL: URL) throws {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let pixelBuffer = try makeZeroPixelBuffer()
    let input = try MLDictionaryFeatureProvider(dictionary: [
        "image": MLFeatureValue(pixelBuffer: pixelBuffer)
    ])
    let prediction = try model.prediction(from: input)
    let outputName = "var_1347"
    guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
        throw WheelsCoreMLSmokeError.missingOutput(outputName)
    }
    guard output.count == 117_600 else {
        throw WheelsCoreMLSmokeError.badOutputCount(output.count)
    }
    print("ok=true output=\(outputName) shape=\(output.shape) count=\(output.count)")
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw WheelsCoreMLSmokeError.usage
    }
    try runSmoke(modelURL: URL(fileURLWithPath: CommandLine.arguments[1]))
} catch {
    fputs("CoreML smoke failed: \(error)\n", stderr)
    exit(1)
}


# Wheels CoreML iOS Decoder Reference

This folder is a reference postprocess handoff for the mobile iOS model.

Use it with:

- model: `best_int8.mlmodel`
- input: CoreML image input named `image`, size `384x384`
- output: `var_1344`
- logical output shape: `[1, 14, 3024]`

The model output is raw YOLO-pose data. The network does not rotate, crop,
raycast, run RANSAC, or return final screen coordinates by itself.

## Files

- `WheelsPoseDecoder.swift` - self-contained Swift decoder.

## Minimal Usage

```swift
let prediction = try model.prediction(from: inputFeatures)

guard let raw = prediction.featureValue(for: "var_1344")?.multiArrayValue else {
    fatalError("Missing CoreML output var_1344")
}

let stats = WheelsPoseDecoder.outputStats(raw)
print("var_1344 stats count=\(stats.count) min=\(stats.min) max=\(stats.max) mean=\(stats.mean) allZero=\(stats.allZero)")

let wheels = try WheelsPoseDecoder.decode(
    output: raw,
    originalSize: CGSize(width: frameWidth, height: frameHeight),
    resizeMode: .stretch,
    confidenceThreshold: 0.05,
    iouThreshold: 0.45,
    maxDetections: 20
)
```

For the first debug run, start with `confidenceThreshold: 0.05`. If detections
appear, raise it to `0.25`.

## Resize Mode

The decoder must match the exact preprocessing used before inference:

- `.stretch` - frame was manually resized directly to `384x384`.
- `.scaleFitLetterbox` - frame was aspect-fit into `384x384` with padding.
- `.centerCrop` - frame was aspect-filled into `384x384` and center-cropped.

If Vision or CoreML applies a different crop/scale/orientation than the app
expects, boxes and keypoints will look shifted even though inference is OK.

The safest integration path is:

1. Rotate the camera frame to the same orientation as the displayed frame.
2. Create an exact `384x384` pixel buffer yourself.
3. Run CoreML directly.
4. Decode `var_1344`.
5. Map decoded bbox/keypoints back to the original frame with the same resize
   mode that was used in step 2.

## Output Contract

`WheelsPoseDecoder.decodePayload(...)` returns the confirmed AR shape:

```json
{
  "frame_id": "frame_001",
  "wheels": [
    {
      "bbox_xyxy": [10.0, 20.0, 100.0, 120.0],
      "confidence": 0.94,
      "points": {
        "a": [20.0, 115.0],
        "b": [90.0, 115.0],
        "c_disc_bottom": [55.0, 96.0]
      }
    }
  ]
}
```

No 3D, no raycast, no plane, no RANSAC, no tracking fields are emitted here.

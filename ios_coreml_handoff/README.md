# iOS CoreML Handoff

This package is the iOS integration handoff for the current Wheels champion
model. It is ready for **integration testing**, not production certification.

## What To Use

Use the exact CoreML artifact:

```text
outputs/production_audit/coreml_export/best.mlmodel
```

Recommended Xcode path:

```text
WheelsApp/Resources/ML/best.mlmodel
```

Xcode will compile it into `best.mlmodelc` during the app build. If the team
wants to load a compiled model manually, use the compiled bundle from the
generated handoff zip:

```text
best.mlmodelc
```

## Model I/O

- Format: CoreML NeuralNetwork `.mlmodel`
- Input count: **1**
- Input name: `image`
- Input type: RGB image
- Input size: `640 x 640`
- Extra `angles_input`: **not used**
- Output count: **1**
- Output name: `var_1347`
- Output logical shape: `[1, 14, 8400]`
- Output value count: `117600`

The TensorFlow snippet that has `input_angles` does not apply to this model.
This model is already exported from the YOLO pose champion and needs only the
image input.

## Runtime Smoke

Copy `WheelsCoreMLSmoke.swift` into an XCTest target or run equivalent code in
the app. It loads `best.mlmodelc`, runs a zero-image smoke inference, and checks
that the model returns a finite output with `117600` values.

The smoke test proves that CoreML can load and execute the packaged artifact.
It does not prove AR quality or production readiness.

## App Integration Notes

Preprocess camera frames as:

1. Convert/crop/resize the camera frame to `640 x 640`.
2. Feed it as CoreML image input named `image`.
3. Read output `var_1347`.
4. Decode YOLO pose raw output as `[1, 14, 8400]`.
5. Use confidence threshold `0.80` for the current audited operating point.
6. Run NMS with IoU `0.45`.
7. Emit the confirmed AR JSON only:

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.94,
      "points": {
        "a": [xa, ya],
        "b": [xb, yb],
        "c_disc_bottom": [xc, yc]
      }
    }
  ]
}
```

Do not emit timestamps, track IDs, visibility flags, per-keypoint confidence,
3D points, raycasts, plane parameters, or RANSAC data from ML. Those belong to
the AR layer.

## Current Certification Scope

Local checks pass:

- CoreML artifact exists and SHA256 is pinned.
- `xcrun coremlcompiler compile` succeeds.
- Desktop package certification is green.

Still missing for production:

- real iPhone/iPad runtime report,
- human-labelled AR-device holdout,
- AR replay/RANSAC validation.

So the handoff status is:

```text
integration_ready=true
production_ready=false
```


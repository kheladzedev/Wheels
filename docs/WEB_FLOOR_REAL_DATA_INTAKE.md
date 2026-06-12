# Web Floor Real Data Intake

The current web-floor model path is ready for real data, but production
training must not start from the fixture. Use this intake shape for the first
real web/phone batch.

## Folder Shape

```text
data/web_floor_real_v1/
  manifest.json
  images/
    frame_0001.jpg
    frame_0002.jpg
```

Config template:

```text
configs/pose_dataset_web_floor_real_template.yaml
```

Copy it to a versioned config before use, for example:

```text
configs/pose_dataset_web_floor_real_v1.yaml
```

## Manifest Shape

```json
{
  "schema": "web_floor_manifest_v1",
  "fixture_only": false,
  "items": [
    {
      "frame_id": "phone-floor-0001",
      "split": "train",
      "image": "images/frame_0001.jpg",
      "provenance": {
        "source": "phone_capture",
        "device": "iPhone 15 Pro",
        "capture_date": "2026-06-12",
        "annotator": "igor"
      },
      "floor": {
        "pitch": -0.04,
        "roll": 0.01,
        "distance": 1.2,
        "distance_mode": "metric_anchor",
        "fov_mode": "provided"
      },
      "wheels": [
        {
          "bbox_xyxy": [120.0, 340.0, 260.0, 520.0],
          "confidence": 1.0,
          "points": {
            "a": [142.0, 505.0],
            "b": [238.0, 506.0],
            "c_disc_bottom": [190.0, 452.0]
          }
        }
      ]
    }
  ]
}
```

## CSV Intake Option

Instead of hand-writing JSON, Igor can send a CSV where one row is one labelled
wheel and frame-level fields repeat for multiple wheels in the same frame.

To generate a ready-to-send request zip with this README, CSV template, CSV
example, and image placeholder, run:

```bash
./.venv/bin/python scripts/create_web_floor_evidence_request_bundle.py
```

This writes:

```text
outputs/web_floor_network/web_floor_real_data_request_bundle.zip
outputs/web_floor_network/web_floor_real_data_request_bundle_manifest.json
```

Required columns:

```text
frame_id,split,image,provenance_source,provenance_device,provenance_annotator,
pitch,roll,distance,distance_mode,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,
a_x,a_y,b_x,b_y,c_disc_bottom_x,c_disc_bottom_y
```

Optional columns:

```text
provenance_capture_date,fov_mode
```

Import command:

```bash
./.venv/bin/python scripts/import_web_floor_annotations.py \
  --csv data/incoming/web_floor_real_v1/annotations.csv \
  --image-root data/incoming/web_floor_real_v1/images \
  --dataset-root data/web_floor_real_v1 \
  --config-out configs/pose_dataset_web_floor_real_v1.yaml \
  --overwrite
```

Then run the gate:

```bash
./.venv/bin/python scripts/audit_web_floor_real_data.py \
  --config configs/pose_dataset_web_floor_real_v1.yaml \
  --output-json outputs/web_floor_network/real_data_gate.json \
  --fail-on-not-ready
```

## Required Fields

- `fixture_only: false`
- `frame_id`: present and unique
- `split`: at least `train` and `holdout`
- `image`: relative path under the dataset root
- `provenance`: non-empty source/device/annotation context
- `floor.pitch`: radians
- `floor.roll`: radians
- `floor.distance`: number in the declared distance mode
- `floor.distance_mode`: not `unknown`
- `wheels[].bbox_xyxy`
- `wheels[].points.a`
- `wheels[].points.b`
- `wheels[].points.c_disc_bottom`

Wheel point semantics stay unchanged:

- `a`: left floor-ray/contact point near the wheel footprint
- `b`: right floor-ray/contact point near the wheel footprint
- `c_disc_bottom`: lowest visible point of the metal disc

## First Batch Minimum

The default gate expects:

- at least 50 frames,
- at least 80 labelled wheels,
- `train` and `holdout` splits,
- provenance for every frame,
- no `unknown` distance mode,
- distance span at least `0.5`,
- pitch or roll span at least `0.05` radians.

Run:

```bash
./.venv/bin/python scripts/audit_web_floor_real_data.py \
  --config configs/pose_dataset_web_floor_real_v1.yaml \
  --output-json outputs/web_floor_network/real_data_gate.json \
  --fail-on-not-ready
```

Only after this passes should we train the production web-floor model.

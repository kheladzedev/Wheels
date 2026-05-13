"""Run the baseline checkpoint over a directory of photos and render a
demo gallery (image + predictions overlaid). Used for presentations
and for sharing the model's current state with the AR team without
asking them to run anything.

Loads the YOLO-pose checkpoint once and reuses it for the whole
directory, which is the only reason this lives here and not in
src/infer_image.py.

Usage:
    python scripts/build_demo_gallery.py \\
        --images-dir data/manual_real/images \\
        --pattern 'real_*.jpg' \\
        --model runs/pose/wheel_baseline_v1/weights/best.pt \\
        --out-dir outputs/demo --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# Reuse the existing overlay so the gallery matches what infer_image.py
# would draw for a single photo.
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer_image import draw_final_overlay, extract_keypoints  # noqa: E402
from postprocess_wheels import build_ar_payload, to_confirmed_schema  # noqa: E402

WHEEL_CLASS_NAMES = {"wheel"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a demo gallery of baseline predictions."
    )
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern under --images-dir (default *.jpg).",
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/demo"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, stop after this many images with at least one detection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise SystemExit(f"images dir not found: {args.images_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "json").mkdir(exist_ok=True)

    images = sorted(args.images_dir.glob(args.pattern))
    if not images:
        raise SystemExit(f"no images match {args.images_dir}/{args.pattern}")

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "pose":
        raise SystemExit(f"model task is {model.task!r}, expected 'pose'")

    summary = []
    kept = 0

    for img_path in images:
        if args.limit and kept >= args.limit:
            break

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        class_names = result.names

        detections = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                conf = float(box.conf.item())
                if conf < args.conf:
                    continue
                cls_id = int(box.cls.item())
                name = class_names.get(cls_id, str(cls_id))
                if WHEEL_CLASS_NAMES and name not in WHEEL_CLASS_NAMES:
                    continue
                kps = extract_keypoints(i, result)
                detections.append(
                    {
                        "class_name": name,
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        "confidence": conf,
                        "keypoints": kps,
                    }
                )

        legacy_payload = build_ar_payload(
            detections,
            conf_threshold=args.conf,
            frame_id=img_path.stem,
            timestamp=0.0,
        )
        confirmed = to_confirmed_schema(legacy_payload)

        n_wheels = len(confirmed["wheels"])
        summary.append({"image": img_path.name, "wheels": n_wheels})

        if n_wheels == 0:
            continue

        original = cv2.imread(str(img_path))
        if original is None:
            print(f"warn: could not decode {img_path}")
            continue

        overlay = draw_final_overlay(original, legacy_payload["wheels"], args.conf)
        out_img = args.out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_img), overlay)

        out_json = args.out_dir / "json" / f"{img_path.stem}.json"
        out_json.write_text(json.dumps(confirmed, indent=2), encoding="utf-8")

        kept += 1
        print(f"[{kept}] {img_path.name} -> {n_wheels} wheel(s)")

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    n_with = sum(1 for s in summary if s["wheels"] > 0)
    n_total_wheels = sum(s["wheels"] for s in summary)
    print()
    print(f"Images processed:     {len(summary)}")
    print(f"Images with wheels:   {n_with}")
    print(f"Total wheels drawn:   {n_total_wheels}")
    print(f"Output dir:           {args.out_dir}")


if __name__ == "__main__":
    main()

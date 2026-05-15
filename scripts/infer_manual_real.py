"""Run a given pose model on the 8 manual_real images and save overlays.

Used by the wheel_v4_real fine-tune /goal to produce before/after
visuals. Ultralytics' save=True writes the overlay JPGs to its global
runs directory; this script copies them back into outputs/<name>/ so
they live alongside the rest of the project artefacts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "manual_real" / "images"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--name", required=True, help="Output subdir under outputs/")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    out_dir = REPO / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    print(f"model: {args.model}  task: {model.task}")

    results = model.predict(
        source=str(SRC),
        conf=args.conf,
        iou=args.iou,
        max_det=20,
        device=args.device,
        save=True,
        project=str(out_dir.parent),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    # Ultralytics may have written into ~/runs/pose/<name>/ instead of
    # outputs/<name>/ depending on its global settings. Copy back.
    ultra_out = Path.home() / "runs" / "pose" / "outputs" / args.name
    if ultra_out.is_dir():
        for f in ultra_out.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copyfile(f, out_dir / f.name)

    total = 0
    counts: dict[str, int] = {}
    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total += n
        name = Path(r.path).name
        counts[name] = n
        print(f"  {name}: {n} wheel(s)")
    counts_path = out_dir / "detection_counts.json"
    counts_path.write_text(
        __import__("json").dumps(counts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\ntotal detections: {total}")
    print(f"overlays in: {out_dir}")
    print(f"counts in: {counts_path}")


if __name__ == "__main__":
    main()

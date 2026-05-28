"""Use wheel_real_v1_clean (high-precision model) to re-predict on real_v1
frames whose original auto-drafts were entirely _needs_review. The original
heuristic was uncertain on those wheels; the trained model has FP rate 0.25,
so high-confidence (>=0.7) predictions on those frames are a reasonable
addition to training data without manual QA.

Output: data/incoming/real_v1_pseudo/{images,annotations,metadata}/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# Local model wrapper
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="runs/pose/wheel_real_v1_clean/weights/best.pt",
        help="Detector to use for pseudo-labeling",
    )
    p.add_argument(
        "--source",
        default="data/incoming/real_v1",
        help="Original bundle (we re-predict on its images)",
    )
    p.add_argument(
        "--output",
        default="data/incoming/real_v1_pseudo",
    )
    p.add_argument("--conf", type=float, default=0.70)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    from ultralytics import YOLO

    src = Path(args.source)
    dst = Path(args.output)
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("images", "annotations", "metadata"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # Frames where the original drafts had no clean wheels (i.e. dropped by
    # filter_clean_real_v1.py). We exclude frames already accepted into
    # real_v1_clean to avoid double-counting.
    clean_root = Path("data/incoming/real_v1_clean/annotations")
    clean_stems: set[str] = (
        {p.stem for p in clean_root.glob("*.json")} if clean_root.is_dir() else set()
    )

    dropped_frames: list[Path] = []
    for jp in sorted((src / "annotations").glob("*.json")):
        if jp.stem in clean_stems:
            continue
        ann = json.loads(jp.read_text())
        wheels = ann.get("wheels", [])
        # Frames whose drafts said either "no wheel" OR "all wheels need review"
        if not wheels or all(w.get("_needs_review") for w in wheels):
            dropped_frames.append(jp)
    print(f"[pseudo] candidate frames: {len(dropped_frames)}")
    if not dropped_frames:
        return 0

    model = YOLO(args.model)

    kept_frames = 0
    kept_wheels = 0
    for jp in dropped_frames:
        ann = json.loads(jp.read_text())
        img_name = ann.get("image") or (jp.stem + ".jpg")
        src_img = src / "images" / img_name
        if not src_img.is_file():
            for ext in (".jpg", ".jpeg", ".png"):
                cand = src / "images" / (jp.stem + ext)
                if cand.is_file():
                    src_img = cand
                    img_name = cand.name
                    break
        if not src_img.is_file():
            continue

        img = cv2.imread(str(src_img))
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model.predict(
            source=str(src_img),
            conf=args.conf,
            iou=0.45,
            max_det=10,
            device=args.device,
            verbose=False,
        )
        if not results:
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        if r.keypoints is None:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kp_xy = r.keypoints.xy.cpu().numpy()  # (N, 3, 2)

        wheels_out = []
        for i in range(len(boxes_xyxy)):
            if float(confs[i]) < args.conf:
                continue
            kp = kp_xy[i]
            if np.any(kp <= 0.5):  # degenerate keypoint
                continue
            wheels_out.append(
                {
                    "bbox_xyxy": [round(float(v), 3) for v in boxes_xyxy[i]],
                    "points": {
                        "a": [round(float(kp[0, 0]), 3), round(float(kp[0, 1]), 3)],
                        "b": [round(float(kp[1, 0]), 3), round(float(kp[1, 1]), 3)],
                        "c_disc_bottom": [
                            round(float(kp[2, 0]), 3),
                            round(float(kp[2, 1]), 3),
                        ],
                    },
                    "_pseudo_conf": round(float(confs[i]), 4),
                }
            )

        if not wheels_out:
            continue

        shutil.copy2(src_img, dst / "images" / img_name)
        (dst / "annotations" / jp.name).write_text(
            json.dumps(
                {
                    "frame_id": ann.get("frame_id") or jp.stem,
                    "image": img_name,
                    "wheels": wheels_out,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        kept_frames += 1
        kept_wheels += len(wheels_out)

    (dst / "metadata" / "source_info.json").write_text(
        json.dumps(
            {
                "source_name": "real_v1_pseudo",
                "annotation_method": (
                    "wheel_real_v1_clean predictions on real_v1 frames that "
                    "had no clean drafts, conf>=%s" % args.conf
                ),
                "_warning": "Pseudo-labels — do not trust without review",
                "kept_frames": kept_frames,
                "kept_wheels": kept_wheels,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[pseudo] kept_frames={kept_frames} kept_wheels={kept_wheels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

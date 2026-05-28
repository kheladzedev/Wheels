"""Use champion soft_s_aug to relabel ALL real_v1 frames at conf ≥ 0.5.

The champion model has Box mAP50 0.814 and FP rate 0.31 — far better than
the original auto_annotate_wheels.py heuristic that generated the drafts
(FP > 0.6 implied by baseline_v1 FP of 0.58). Re-labelling at conf ≥ 0.5
should give us a cleaner-than-original ground truth approximation, plus
more wheels than the soft filter alone.

Output: data/incoming/real_v1_self/{images,annotations,metadata}/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt",
    )
    p.add_argument(
        "--source",
        default="data/incoming/real_v1",
    )
    p.add_argument(
        "--output",
        default="data/incoming/real_v1_self",
    )
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    from ultralytics import YOLO

    src = Path(args.source)
    dst = Path(args.output)
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("images", "annotations", "metadata"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    frames = sorted((src / "annotations").glob("*.json"))
    print(f"[self] {len(frames)} candidate frames")

    kept_frames = 0
    kept_wheels = 0
    for jp in frames:
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
            max_det=20,
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
        kp_xy = r.keypoints.xy.cpu().numpy()

        wheels_out = []
        for i in range(len(boxes_xyxy)):
            if float(confs[i]) < args.conf:
                continue
            kp = kp_xy[i]
            if np.any(kp <= 0.5):
                continue
            # Sanity: each keypoint inside image
            if (
                kp[:, 0].min() < 0
                or kp[:, 0].max() >= w
                or kp[:, 1].min() < 0
                or kp[:, 1].max() >= h
            ):
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
                "source_name": "real_v1_self",
                "annotation_method": f"champion soft_s_aug self-predict on real_v1, conf≥{args.conf}",
                "model": args.model,
                "kept_frames": kept_frames,
                "kept_wheels": kept_wheels,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[self] kept_frames={kept_frames} kept_wheels={kept_wheels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

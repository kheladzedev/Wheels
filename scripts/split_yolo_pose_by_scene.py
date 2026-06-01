"""Re-split a YOLO-pose dataset by scene_id so adjacent capture frames cannot
leak between train and val.

Reads `metadata/scene_ids.json` from the *incoming* root (written by
`scripts/assign_scene_ids.py`) plus the converted dataset's
`metadata/split_manifest.json`, then moves files between `images/{train,val}`
and `labels/{train,val}` so val contains only the requested val_scenes.

Usage:
    python scripts/split_yolo_pose_by_scene.py \\
        --dataset-root outputs/unreal_export_acceptance_neuraldata1/<slug>/pose_dataset \\
        --incoming-root outputs/raw_unreal_exports/<slug> \\
        --val-scenes 8,9
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _move(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--incoming-root", type=Path, required=True)
    p.add_argument(
        "--val-scenes", default="8,9", help="comma-separated scene_ids that go into val"
    )
    args = p.parse_args()

    val_scenes = {int(s) for s in args.val_scenes.split(",") if s.strip()}
    scene_path = args.incoming_root / "metadata" / "scene_ids.json"
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene_ids.json missing: {scene_path}")
    scene_doc = json.loads(scene_path.read_text())
    frame_to_scene = {int(k): int(v) for k, v in scene_doc["frame_to_scene"].items()}

    images_train = args.dataset_root / "images" / "train"
    images_val = args.dataset_root / "images" / "val"
    labels_train = args.dataset_root / "labels" / "train"
    labels_val = args.dataset_root / "labels" / "val"

    moved_to_val = 0
    moved_to_train = 0

    _IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    # Iterate over union of train + val image stems.
    def _all_image_files() -> list[Path]:
        out = []
        for d in (images_train, images_val):
            if d.is_dir():
                for ext in _IMAGE_EXTS:
                    out.extend(d.glob(ext))
        return sorted(set(out))

    for img_path in _all_image_files():
        stem = img_path.stem
        # stem format: "<source_slug>__<frame_idx>"
        try:
            frame_idx = int(stem.rsplit("__", 1)[-1])
        except ValueError:
            continue
        target_scene = frame_to_scene.get(frame_idx)
        if target_scene is None:
            continue
        currently_val = img_path.parent.name == "val"
        wants_val = target_scene in val_scenes
        if wants_val == currently_val:
            continue
        if wants_val:
            _move(img_path, images_val / img_path.name)
            _move(labels_train / (stem + ".txt"), labels_val / (stem + ".txt"))
            moved_to_val += 1
        else:
            _move(img_path, images_train / img_path.name)
            _move(labels_val / (stem + ".txt"), labels_train / (stem + ".txt"))
            moved_to_train += 1

    summary = {
        "val_scenes": sorted(val_scenes),
        "moved_to_val": moved_to_val,
        "moved_to_train": moved_to_train,
        "final_counts": {
            "train_images": sum(
                len(list(images_train.glob(ext))) for ext in _IMAGE_EXTS
            ),
            "val_images": sum(len(list(images_val.glob(ext))) for ext in _IMAGE_EXTS),
            "train_labels": len(list(labels_train.glob("*.txt"))),
            "val_labels": len(list(labels_val.glob("*.txt"))),
        },
    }
    meta_dir = args.dataset_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "scene_aware_split.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()

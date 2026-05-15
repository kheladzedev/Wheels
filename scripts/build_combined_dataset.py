"""Combine the cartoon wheel_dataset with the real seed batch.

Layout:
    data/wheel_combined_dataset/
      images/train/  — symlinks to wheel_dataset/train + N copies of real train
      images/val/    — symlinks to wheel_dataset/val   + real val (1 copy)
      labels/train/  — corresponding .txt files
      labels/val/    — corresponding .txt files

Real images are physically duplicated (filename_dupK suffix) so Ultralytics
mosaic + augment pipeline samples each real frame proportionally to the
desired weight. With 6 cartoon-heavy real seeds vs 240 cartoon train
samples, we duplicate the real images ×REAL_DUPE so the per-epoch ratio
of real:cartoon is roughly 1:2 — enough to push gradients toward the
real-photo distribution without losing the diverse-pose prior cartoons
provide.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARTOON = REPO / "data" / "wheel_dataset"
REAL = REPO / "data" / "wheel_pose_dataset_real"
OUT = REPO / "data" / "wheel_combined_dataset"

REAL_DUPE_TRAIN = 20  # 6 train * 20 = 120 vs 240 cartoon → ~1:2
REAL_DUPE_VAL = 1


def _link_one(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _copy_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> int:
    n = 0
    for img in sorted(src_img.iterdir()):
        if img.is_file():
            _link_one(img, dst_img / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.exists():
                _link_one(lbl, dst_lbl / lbl.name)
            n += 1
    return n


def _duplicate_real(
    src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path, k: int
) -> int:
    n = 0
    for img in sorted(src_img.iterdir()):
        if not img.is_file():
            continue
        lbl = src_lbl / f"{img.stem}.txt"
        for i in range(k):
            dst_i = dst_img / f"{img.stem}__dup{i:02d}{img.suffix}"
            dst_l = dst_lbl / f"{img.stem}__dup{i:02d}.txt"
            _link_one(img, dst_i)
            if lbl.exists():
                _link_one(lbl, dst_l)
            n += 1
    return n


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    cart_train = _copy_split(
        CARTOON / "images" / "train",
        CARTOON / "labels" / "train",
        OUT / "images" / "train",
        OUT / "labels" / "train",
    )
    cart_val = _copy_split(
        CARTOON / "images" / "val",
        CARTOON / "labels" / "val",
        OUT / "images" / "val",
        OUT / "labels" / "val",
    )
    real_train = _duplicate_real(
        REAL / "images" / "train",
        REAL / "labels" / "train",
        OUT / "images" / "train",
        OUT / "labels" / "train",
        REAL_DUPE_TRAIN,
    )
    real_val = _duplicate_real(
        REAL / "images" / "val",
        REAL / "labels" / "val",
        OUT / "images" / "val",
        OUT / "labels" / "val",
        REAL_DUPE_VAL,
    )

    print(f"cartoon train: {cart_train}, real train (xdupe): {real_train}")
    print(f"cartoon val:   {cart_val}, real val   (xdupe): {real_val}")
    print(f"Combined dataset at: {OUT}")


if __name__ == "__main__":
    main()

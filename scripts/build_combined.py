"""Merge real_v1_soft + real_v1_pseudo into a single incoming bundle.

Both are plugin-format already (frame_id, image, wheels[].{bbox_xyxy, points})
so we just copy everything into one directory, taking care that file names
don't collide (real_v1 stems are unique across both filters so no rename).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SOURCES = [
    Path("data/incoming/real_v1_soft"),
    Path("data/incoming/real_v1_pseudo"),
]
DST = Path("data/incoming/real_v1_combined")


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images", "annotations", "metadata"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_wheels = 0
    seen_stems: set[str] = set()

    for src in SOURCES:
        if not src.is_dir():
            print(f"[combined] missing {src}, skipping")
            continue
        for jp in sorted((src / "annotations").glob("*.json")):
            stem = jp.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            ann = json.loads(jp.read_text())
            img_name = ann.get("image") or (stem + ".jpg")
            src_img = src / "images" / img_name
            if not src_img.is_file():
                continue
            shutil.copy2(src_img, DST / "images" / img_name)
            (DST / "annotations" / jp.name).write_text(
                json.dumps(ann, indent=2, ensure_ascii=False)
            )
            total_frames += 1
            total_wheels += len(ann.get("wheels", []))

    (DST / "metadata" / "source_info.json").write_text(
        json.dumps(
            {
                "source_name": "real_v1_combined",
                "annotation_method": "real_v1_soft + real_v1_pseudo merged",
                "components": [str(s) for s in SOURCES],
                "total_frames": total_frames,
                "total_wheels": total_wheels,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[combined] frames={total_frames} wheels={total_wheels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

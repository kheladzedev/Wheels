"""Build a 'clean-only' incoming bundle from real_v1 by dropping every wheel
flagged _needs_review. The hypothesis is that the auto-draft heuristic is
correct on its high-confidence ~40 % subset and the rest poisons training.

Output: data/incoming/real_v1_clean/{images,annotations,metadata}/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("data/incoming/real_v1")
DST = Path("data/incoming/real_v1_clean")


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images", "annotations", "metadata"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    kept_frames = 0
    dropped_frames = 0
    kept_wheels = 0
    dropped_wheels = 0
    for jp in sorted((SRC / "annotations").glob("*.json")):
        ann = json.loads(jp.read_text())
        wheels = ann.get("wheels", [])
        clean = [w for w in wheels if not w.get("_needs_review")]
        dropped_wheels += len(wheels) - len(clean)
        if not clean:
            dropped_frames += 1
            continue
        # write a new annotation with only clean wheels; clear draft flags so
        # the converter accepts it
        new_ann = dict(ann)
        new_ann["wheels"] = clean
        new_ann.pop("_draft", None)
        new_ann.pop("_warning", None)
        for w in new_ann["wheels"]:
            w.pop("_needs_review", None)
            w.pop("_review_reasons", None)
        # copy image
        img_name = ann.get("image")
        if img_name is None:
            img_name = jp.stem + ".jpg"
        src_img = SRC / "images" / img_name
        if not src_img.is_file():
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = SRC / "images" / (jp.stem + ext)
                if candidate.is_file():
                    src_img = candidate
                    img_name = candidate.name
                    break
        if not src_img.is_file():
            continue
        shutil.copy2(src_img, DST / "images" / img_name)
        new_ann["image"] = img_name
        (DST / "annotations" / jp.name).write_text(
            json.dumps(new_ann, indent=2, ensure_ascii=False)
        )
        kept_frames += 1
        kept_wheels += len(clean)

    info = {
        "source_name": "real_v1_clean",
        "annotation_method": "real_v1 auto_annotate_wheels.py drafts, _needs_review wheels dropped",
        "_warning": "Filtered subset — high-confidence drafts only, no human review yet",
        "kept_frames": kept_frames,
        "dropped_frames": dropped_frames,
        "kept_wheels": kept_wheels,
        "dropped_wheels": dropped_wheels,
    }
    (DST / "metadata" / "source_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )
    print(
        f"[filter] kept frames={kept_frames} (dropped {dropped_frames}); "
        f"wheels={kept_wheels} (dropped {dropped_wheels})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Soft-filter real_v1: drop wheel only if it has >=2 review reasons.

A wheel with a single soft flag (e.g. low_circularity alone) is usually still
broadly correct — the heuristic just had a moderate doubt. Wheels with two or
more concerns (small bbox AND low circularity, etc.) are the ones most likely
to be wrong about A/B/C placement. Keeping the single-flag wheels recovers
roughly +30 wheels vs the strict 'clean' filter.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("data/incoming/real_v1")
DST = Path("data/incoming/real_v1_soft")
MAX_REVIEW_REASONS = 1  # keep wheels with 0 or 1 review reasons


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images", "annotations", "metadata"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    kept_frames = dropped_frames = kept_wheels = dropped_wheels = 0
    for jp in sorted((SRC / "annotations").glob("*.json")):
        ann = json.loads(jp.read_text())
        wheels = ann.get("wheels", [])
        kept = [
            w
            for w in wheels
            if len(w.get("_review_reasons", []) or []) <= MAX_REVIEW_REASONS
        ]
        dropped_wheels += len(wheels) - len(kept)
        if not kept:
            dropped_frames += 1
            continue
        new_ann = dict(ann)
        new_ann["wheels"] = kept
        new_ann.pop("_draft", None)
        new_ann.pop("_warning", None)
        for w in new_ann["wheels"]:
            w.pop("_needs_review", None)
            w.pop("_review_reasons", None)
        img_name = ann.get("image") or (jp.stem + ".jpg")
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
        kept_wheels += len(kept)

    (DST / "metadata" / "source_info.json").write_text(
        json.dumps(
            {
                "source_name": "real_v1_soft",
                "annotation_method": f"real_v1 auto-drafts, drop wheels with >{MAX_REVIEW_REASONS} review reasons",
                "kept_frames": kept_frames,
                "kept_wheels": kept_wheels,
                "dropped_wheels": dropped_wheels,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"[soft] kept frames={kept_frames} dropped={dropped_frames} "
        f"wheels={kept_wheels} dropped={dropped_wheels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

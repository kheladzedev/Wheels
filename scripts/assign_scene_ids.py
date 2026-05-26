"""Post-process: assign scene_id per frame based on capture run boundaries.

The CameraCaptureWheels spline yields ~N consecutive frames per run (default
100), and each run gets a freshly randomised camera rotation. Frames within
one run share the same camera trajectory and are therefore highly correlated;
frames across runs are independent. Without explicit scene IDs the converter
splits frames randomly and may leak a run between train/val.

This script reads an incoming dataset (the one accept_neuraldata1_capture.py
copied into outputs/raw_unreal_exports/<slug>/), groups frame indices into
runs by sequential numbering, and writes an Igor-equivalent scene_id field
into each annotation JSON.

Usage:
    python scripts/assign_scene_ids.py \\
        --incoming-root outputs/raw_unreal_exports/<slug> \\
        --frames-per-run 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-root", type=Path, required=True)
    parser.add_argument("--frames-per-run", type=int, default=100)
    parser.add_argument("--annotations-subdir", default="annotations")
    parser.add_argument("--metadata-subdir", default="metadata")
    args = parser.parse_args()

    ann_dir = args.incoming_root / args.annotations_subdir
    if not ann_dir.is_dir():
        # Some intake layouts use keyPoint/ instead of annotations/.
        for cand in ("keyPoint", "annotations_drafts"):
            alt = args.incoming_root / cand
            if alt.is_dir():
                ann_dir = alt
                break
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"no annotations dir under {args.incoming_root}")

    # Discover frame indices from JSON file stems if they parse as int,
    # otherwise from sibling images/ folder.
    indices: list[int] = []
    parser_kind: str
    json_files = sorted(p for p in ann_dir.glob("*.json"))
    if json_files:
        for p in json_files:
            try:
                indices.append(int(p.stem))
            except ValueError:
                pass
    if indices:
        parser_kind = "annotation_stems"
    else:
        # Fallback: enumerate sibling images.
        images_dir = args.incoming_root / "images"
        if images_dir.is_dir():
            for p in sorted(images_dir.glob("*.jpg")):
                try:
                    indices.append(int(p.stem))
                except ValueError:
                    pass
            parser_kind = "image_stems"
        else:
            parser_kind = "none"

    if not indices:
        raise RuntimeError("could not derive frame indices from incoming")
    indices.sort()

    base = indices[0]
    span = args.frames_per_run
    mapping: dict[int, int] = {}
    for idx in indices:
        scene_id = (idx - base) // span
        mapping[idx] = scene_id

    # Write per-frame mapping to metadata.
    meta_dir = args.incoming_root / args.metadata_subdir
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_map = meta_dir / "scene_ids.json"
    out_map.write_text(
        json.dumps(
            {
                "frames_per_run": span,
                "base_index": base,
                "indices_source": parser_kind,
                "frame_to_scene": {str(k): v for k, v in sorted(mapping.items())},
                "scene_counts": {
                    str(s): sum(1 for v in mapping.values() if v == s)
                    for s in sorted(set(mapping.values()))
                },
            },
            indent=2,
        )
    )

    # Patch annotation JSONs in place so the converter can pick it up.
    patched = 0
    for p in json_files:
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        if idx not in mapping:
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        doc["scene_id"] = mapping[idx]
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        patched += 1

    print(
        f"scene_ids assigned: frames={len(mapping)}, scenes={len(set(mapping.values()))}, patched_annotations={patched}"
    )
    print(f"mapping written: {out_map}")


if __name__ == "__main__":
    main()

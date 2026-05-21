"""Batch YOLO-pose inference for AR RANSAC calibration.

The AR client wants a deterministic, replayable stream of per-frame JSON
payloads so it can tune RANSAC parameters offline against a recorded
session. ``infer_image.py`` only handles a single image at a time. This
script accepts either a directory of images or a single video file and
emits one AR-contract JSON per frame, plus a ``batch_summary.json``.

Per the AR-spec PDF (page 9, "Механика «Примерка колес»"):

  > Чтобы корректно подобрать ransac параметры, мне нужен лог попаданий,
  > чтоб понять как сильно все шумит изза трекинга и какова стабильность.

We do NOT do 3D, RANSAC, plane reconstruction, or track-id assignment —
that's the AR client's job. We only replay the model and dump the
per-frame contract.

Schema policy (mirrors ``infer_image.py``):

  Primary output is the **AR-team confirmed schema** — exactly
  ``{frame_id, wheels[].{bbox_xyxy, confidence, points.{a, b,
  c_disc_bottom}}}``. No image paths, no thresholds, no timestamps,
  no per-keypoint visibility. The legacy intermediate payload (with
  per-keypoint visibility, warnings, stats, image meta) is written
  only when ``--emit-legacy`` is passed, with a ``_legacy.json``
  suffix. The historic pre-confirmed "target" draft schema is no
  longer accepted or written by this script.

Usage:
    python src/infer_batch.py \\
        --source data/wheel_dataset/images/val \\
        --model runs/pose/wheel_v3/weights/best.pt \\
        --out-dir /tmp/batch_out \\
        --device cpu --max-frames 3

Output layout (per-frame mode, default):
    <out-dir>/<stem>__frame_000000.json         AR confirmed schema (primary)
    <out-dir>/<stem>__frame_000000_legacy.json  if --emit-legacy
    <out-dir>/batch_summary.json                totals + frame index

Combined-jsonl mode (--combined-jsonl):
    <out-dir>/<stem>.jsonl                      one confirmed payload per line
    <out-dir>/<stem>_legacy.jsonl               if --emit-legacy
    <out-dir>/batch_summary.json

Note: ``batch_summary.json`` includes a ``frame_index`` array (per-frame
manifest of frame_id + JSON path) as an extension beyond the spec's 14
keys, so AR can correlate frames to outputs without scanning out_dir.
The ``json`` field in each entry points to the **confirmed primary**;
``legacy_json`` is populated only when --emit-legacy is on, else null.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from postprocess_wheels import (
    N_KEYPOINTS,
    assert_confirmed_no_forbidden_fields,
    build_ar_payload,
    to_confirmed_schema,
    visibility_from_keypoint_confidence,
)

# Keys that must NEVER appear in the confirmed-schema payload that AR
# receives — enforced before writing each frame. Mirrors the guard
# used by infer_image.py.
CONFIRMED_FORBIDDEN_TOP_LEVEL = (
    "image",
    "image_size",
    "thresholds",
    "stats",
    "timestamp",
    "track_id",
)
CONFIRMED_FORBIDDEN_WHEEL_KEYS = (
    "wheel_bbox",
    "bbox_xywh",
    "keypoints",
    "keypoints_confidence",
    "points_confidence",
    "visibility",
    "warnings",
    "track_id",
)

WHEEL_CLASS_NAMES = {"wheel"}
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

# Default fps assumed for directory-of-images sources when --fps is not
# passed. 30 is a reasonable AR-session capture rate; AR can override
# via --fps if their recording was at a different cadence.
DEFAULT_IMAGE_FPS = 30.0


# ---------------------------------------------------------------------------
# Pure helpers (testable without YOLO / cv2)
# ---------------------------------------------------------------------------


def resolve_source_type(path: Path) -> Literal["directory", "video", "unknown"]:
    """Classify a source path as ``directory``, ``video``, or ``unknown``.

    A path is ``directory`` if it exists and is a directory; ``video`` if it
    is a file whose lowercase suffix is in ``VIDEO_EXTS``; ``unknown`` for
    anything else (missing, regular file with an unknown suffix, etc.).
    """
    if path.is_dir():
        return "directory"
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return "video"
    return "unknown"


def frame_id_for_image(image_stem: str) -> str:
    """Build the ``frame_id`` for a directory-of-images source.

    Currently the stem is passed through unchanged — AR captures usually
    save with stable, monotonic file names (e.g. ``sample_0007``) that are
    already a fine frame ID. The function is kept as a seam so the
    convention can evolve (e.g. prefixing with a session ID) without
    rewriting every call site.
    """
    return image_stem


def frame_id_for_video(video_stem: str, frame_index: int) -> str:
    """Build the ``frame_id`` for a video-file source.

    Format: ``<video_stem>_frame_<i:06d>`` where ``i`` is the ORIGINAL
    frame index in the video (not the post-subsample position). This lets
    AR pair the JSON back to the camera transform it logged at the same
    real frame, even when ``--every-n-frames`` skipped most of them.
    """
    return f"{video_stem}_frame_{frame_index:06d}"


def timestamp_for_video(frame_index: int, fps: float) -> float:
    """Compute the timestamp (seconds from video start) for a video frame.

    ``frame_index`` is the original index, ``fps`` comes from the video
    metadata (or ``--fps`` override). Raises ``ValueError`` on non-positive
    fps because the division would otherwise produce ``inf`` or a
    misleading negative value.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return frame_index / fps


def timestamp_for_image_index(index: int, fps: float = DEFAULT_IMAGE_FPS) -> float:
    """Compute a synthetic timestamp for a frame in a directory listing.

    Directory sources don't carry real wall-clock times, so we fabricate a
    monotonic timestamp at ``DEFAULT_IMAGE_FPS`` (30 fps) by default. AR
    can override with ``--fps`` if their session was recorded at a
    different rate. Raises ``ValueError`` on non-positive fps.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return index / fps


def iter_image_paths(
    directory: Path, extensions: tuple[str, ...] = IMAGE_EXTS
) -> list[Path]:
    """Return a deterministically sorted list of image paths in ``directory``.

    Suffix matching is case-insensitive (so ``IMG_001.JPG`` is included).
    The order is plain lexicographic sort over the full path strings,
    which gives a stable, reproducible iteration order across runs.
    """
    exts_lower = {e.lower() for e in extensions}
    out = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts_lower
    ]
    out.sort()
    return out


# ---------------------------------------------------------------------------
# YOLO-touching glue (only used inside main())
# ---------------------------------------------------------------------------


def extract_keypoints(box_idx: int, result) -> list[dict]:
    """Extract 3 keypoints for the i-th detection from a YOLO-pose result.

    Mirrors ``infer_image.extract_keypoints`` — copied rather than imported
    because ``infer_image`` is a script with import-time side effects.

    Each entry: ``{"xy": [x, y], "visibility": int, "confidence": float}``.
    Visibility is inferred from per-keypoint confidence (Ultralytics
    doesn't emit a separate visibility flag at inference time).
    """
    if result.keypoints is None:
        return []
    xy = result.keypoints.xy[box_idx].cpu().numpy()
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[box_idx].cpu().numpy()
    else:
        conf = [1.0] * xy.shape[0]

    kps: list[dict] = []
    for i in range(xy.shape[0]):
        c = float(conf[i])
        vis = visibility_from_keypoint_confidence(c)
        kps.append(
            {
                "xy": [float(xy[i, 0]), float(xy[i, 1])],
                "visibility": vis,
                "confidence": c,
            }
        )
    return kps


def detections_from_result(result, conf_threshold: float, max_det: int) -> list[dict]:
    """Convert one Ultralytics result into the flat-detection-list shape.

    Defence-in-depth filter against YOLO's own ``conf=`` and ``max_det=``
    arguments: we re-filter here so a regression in Ultralytics defaults
    can't leak low-confidence garbage into the AR payload.
    """
    class_names: dict[int, str] = result.names
    model_has_wheels = any(name in WHEEL_CLASS_NAMES for name in class_names.values())

    detections: list[dict] = []
    if result.boxes is None:
        return detections

    for i, box in enumerate(result.boxes):
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls.item())
        name = class_names.get(cls_id, str(cls_id))
        if model_has_wheels and name not in WHEEL_CLASS_NAMES:
            continue
        kps = extract_keypoints(i, result)
        if kps and len(kps) != N_KEYPOINTS:
            print(
                f"WARNING: model emitted {len(kps)} keypoints, expected {N_KEYPOINTS}. "
                "Dropping detection."
            )
            continue
        detections.append(
            {
                "class_name": name,
                "bbox": [float(v) for v in box.xyxy[0].tolist()],
                "confidence": conf,
                "keypoints": kps,
            }
        )

    if len(detections) > max_det:
        # Keep the top-K by confidence as a final safety net.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        detections = detections[:max_det]
    return detections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch YOLO-pose inference (image dir or video file)"
    )
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to a directory of images OR a single video file.",
    )
    p.add_argument(
        "--model", required=True, type=Path, help="Path to YOLO-pose weights (.pt)."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for per-frame JSON and batch_summary.json.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: 'mps' on Apple Silicon, 'cpu', or '0' for CUDA.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument(
        "--max-det", type=int, default=20, help="Hard cap on detections per frame."
    )
    p.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="Subsample: process every N-th frame (default 1 = every frame).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Hard cap on frames inferred (post-subsample). Useful for smoke tests.",
    )
    p.add_argument(
        "--combined-jsonl",
        action="store_true",
        help="Write one combined .jsonl per output instead of one JSON per frame.",
    )
    p.add_argument(
        "--emit-legacy",
        action="store_true",
        help=(
            "Additionally emit the legacy intermediate payload (with "
            "wheel_bbox/keypoints/visibility/warnings/stats + image meta) "
            "as <stem>__frame_XXX_legacy.json. Useful for debugging; AR "
            "never reads it. The primary <stem>__frame_XXX.json always "
            "uses the confirmed AR contract."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Override fps for timestamps. For video, defaults to "
            "VideoCapture.get(CAP_PROP_FPS). For directory, defaults to "
            f"{DEFAULT_IMAGE_FPS}."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference drivers
# ---------------------------------------------------------------------------


def _assert_no_forbidden(confirmed: dict, source_label: str) -> None:
    """Raise if any banned field leaked into the confirmed primary payload.

    Run on every frame just before writing — cheap, catches contract drift
    introduced by upstream changes to ``to_confirmed_schema`` or to this
    script. The guards mirror infer_image.py.
    """
    assert_confirmed_no_forbidden_fields(confirmed, source_label=source_label)
    leaked_top = [k for k in CONFIRMED_FORBIDDEN_TOP_LEVEL if k in confirmed]
    if leaked_top:
        raise AssertionError(
            f"{source_label}: forbidden top-level keys in confirmed payload: "
            f"{leaked_top}"
        )
    for i, w in enumerate(confirmed.get("wheels", [])):
        leaked = [k for k in CONFIRMED_FORBIDDEN_WHEEL_KEYS if k in w]
        if leaked:
            raise AssertionError(
                f"{source_label}: wheel[{i}] in confirmed payload has forbidden "
                f"keys: {leaked}"
            )


def _build_payloads(
    detections: list[dict],
    *,
    conf: float,
    frame_id: str,
    timestamp: float,
    img_size: list[int],
    thresholds: dict,
    image_field: str,
    want_legacy: bool,
) -> tuple[dict, dict | None]:
    """Build the AR-confirmed primary payload plus optional legacy companion.

    Returns ``(confirmed, legacy_or_None)``. The confirmed payload is what
    AR consumes — strictly ``{frame_id, wheels[].{bbox_xyxy, confidence,
    points}}``. The legacy companion (when requested) carries the
    intermediate per-keypoint visibility / warnings / stats / image meta
    used internally for debug overlays and was the old primary shape; it
    is never read by AR.
    """
    legacy_payload = build_ar_payload(
        detections, conf_threshold=conf, frame_id=frame_id, timestamp=timestamp
    )
    confirmed_payload = to_confirmed_schema(legacy_payload)
    _assert_no_forbidden(confirmed_payload, source_label=f"frame {frame_id}")

    if not want_legacy:
        return confirmed_payload, None

    legacy_with_meta = dict(legacy_payload)
    legacy_with_meta["image"] = image_field
    legacy_with_meta["image_size"] = img_size
    legacy_with_meta["thresholds"] = thresholds
    return confirmed_payload, legacy_with_meta


def _write_per_frame(
    out_dir: Path,
    stem: str,
    frame_index: int,
    confirmed_payload: dict,
    legacy_payload: dict | None,
) -> tuple[Path, Path | None]:
    """Persist one frame.

    Primary: ``<stem>__frame_<i:06d>.json`` — confirmed AR schema.
    Optional: ``<stem>__frame_<i:06d>_legacy.json`` — legacy intermediate
    payload when --emit-legacy was passed. AR never reads the legacy file;
    it exists for ML-side debugging only.
    """
    base = out_dir / f"{stem}__frame_{frame_index:06d}"
    json_path = base.with_suffix(".json")
    json_path.write_text(
        json.dumps(confirmed_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    legacy_path: Path | None = None
    if legacy_payload is not None:
        legacy_path = out_dir / f"{stem}__frame_{frame_index:06d}_legacy.json"
        legacy_path.write_text(
            json.dumps(legacy_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return json_path, legacy_path


def _run_directory(
    model,
    *,
    source: Path,
    args: argparse.Namespace,
    out_dir: Path,
    stem: str,
) -> dict:
    """Run inference over a directory of images. Returns batch stats."""
    image_paths = iter_image_paths(source)
    frames_seen = len(image_paths)
    fps_used = args.fps if args.fps is not None else DEFAULT_IMAGE_FPS
    thresholds = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det}

    frames_inferred = 0
    wheels_detected_total = 0
    frame_index_list: list[dict] = []
    want_legacy = bool(args.emit_legacy)

    jsonl_fh = None
    jsonl_legacy_fh = None
    if args.combined_jsonl:
        jsonl_fh = (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8")
        if want_legacy:
            jsonl_legacy_fh = (out_dir / f"{stem}_legacy.jsonl").open(
                "w", encoding="utf-8"
            )

    try:
        for original_idx, img_path in enumerate(image_paths):
            if args.every_n_frames > 1 and original_idx % args.every_n_frames != 0:
                continue
            if args.max_frames is not None and frames_inferred >= args.max_frames:
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
            detections = detections_from_result(result, args.conf, args.max_det)

            img_size = [int(result.orig_shape[1]), int(result.orig_shape[0])]
            frame_id = frame_id_for_image(img_path.stem)
            timestamp = timestamp_for_image_index(original_idx, fps=fps_used)

            confirmed_payload, legacy_payload = _build_payloads(
                detections,
                conf=args.conf,
                frame_id=frame_id,
                timestamp=timestamp,
                img_size=img_size,
                thresholds=thresholds,
                image_field=str(img_path),
                want_legacy=want_legacy,
            )

            n_wheels = len(confirmed_payload["wheels"])
            wheels_detected_total += n_wheels
            frames_inferred += 1

            if args.combined_jsonl:
                if jsonl_fh is None:
                    raise RuntimeError(
                        "jsonl_fh is None inside combined-jsonl branch — bug"
                    )
                jsonl_fh.write(json.dumps(confirmed_payload, ensure_ascii=False) + "\n")
                if jsonl_legacy_fh is not None and legacy_payload is not None:
                    jsonl_legacy_fh.write(
                        json.dumps(legacy_payload, ensure_ascii=False) + "\n"
                    )
                frame_index_list.append(
                    {
                        "frame_id": frame_id,
                        "source_image": str(img_path),
                        "n_wheels": n_wheels,
                    }
                )
            else:
                json_path, legacy_path = _write_per_frame(
                    out_dir, stem, original_idx, confirmed_payload, legacy_payload
                )
                frame_index_list.append(
                    {
                        "frame_id": frame_id,
                        "source_image": str(img_path),
                        "n_wheels": n_wheels,
                        "json": str(json_path),
                        "legacy_json": str(legacy_path) if legacy_path else None,
                    }
                )
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()
        if jsonl_legacy_fh is not None:
            jsonl_legacy_fh.close()

    return {
        "fps_used": fps_used,
        "frames_seen": frames_seen,
        "frames_inferred": frames_inferred,
        "wheels_detected_total": wheels_detected_total,
        "frame_index": frame_index_list,
    }


def _run_video(
    model,
    *,
    source: Path,
    args: argparse.Namespace,
    out_dir: Path,
    stem: str,
) -> dict:
    """Run inference over a single video file. Returns batch stats."""
    import cv2  # local import — keep CLI helpers cv2-free

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")

    frames_seen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_meta = float(cap.get(cv2.CAP_PROP_FPS))
    fps_used = args.fps if args.fps is not None else fps_meta
    if fps_used <= 0:
        cap.release()
        raise RuntimeError(
            f"Video fps is {fps_meta}; pass --fps to override (cannot timestamp frames)."
        )
    thresholds = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det}

    frames_inferred = 0
    wheels_detected_total = 0
    frame_index_list: list[dict] = []
    want_legacy = bool(args.emit_legacy)

    jsonl_fh = None
    jsonl_legacy_fh = None
    if args.combined_jsonl:
        jsonl_fh = (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8")
        if want_legacy:
            jsonl_legacy_fh = (out_dir / f"{stem}_legacy.jsonl").open(
                "w", encoding="utf-8"
            )

    try:
        original_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            try:
                if args.every_n_frames > 1 and original_idx % args.every_n_frames != 0:
                    continue
                if args.max_frames is not None and frames_inferred >= args.max_frames:
                    break

                # YOLO accepts ndarray frames directly.
                results = model.predict(
                    source=frame_bgr,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    verbose=False,
                )
                result = results[0]
                detections = detections_from_result(result, args.conf, args.max_det)

                img_size = [int(result.orig_shape[1]), int(result.orig_shape[0])]
                frame_id = frame_id_for_video(stem, original_idx)
                timestamp = timestamp_for_video(original_idx, fps=fps_used)

                confirmed_payload, legacy_payload = _build_payloads(
                    detections,
                    conf=args.conf,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    img_size=img_size,
                    thresholds=thresholds,
                    image_field=f"{source}#frame={original_idx}",
                    want_legacy=want_legacy,
                )

                n_wheels = len(confirmed_payload["wheels"])
                wheels_detected_total += n_wheels
                frames_inferred += 1

                if args.combined_jsonl:
                    if jsonl_fh is None:
                        raise RuntimeError(
                            "jsonl_fh is None inside combined-jsonl branch — bug"
                        )
                    jsonl_fh.write(
                        json.dumps(confirmed_payload, ensure_ascii=False) + "\n"
                    )
                    if jsonl_legacy_fh is not None and legacy_payload is not None:
                        jsonl_legacy_fh.write(
                            json.dumps(legacy_payload, ensure_ascii=False) + "\n"
                        )
                    frame_index_list.append(
                        {
                            "frame_id": frame_id,
                            "original_frame_index": original_idx,
                            "n_wheels": n_wheels,
                        }
                    )
                else:
                    json_path, legacy_path = _write_per_frame(
                        out_dir, stem, original_idx, confirmed_payload, legacy_payload
                    )
                    frame_index_list.append(
                        {
                            "frame_id": frame_id,
                            "original_frame_index": original_idx,
                            "n_wheels": n_wheels,
                            "json": str(json_path),
                            "legacy_json": str(legacy_path) if legacy_path else None,
                        }
                    )
            finally:
                original_idx += 1
    finally:
        cap.release()
        if jsonl_fh is not None:
            jsonl_fh.close()
        if jsonl_legacy_fh is not None:
            jsonl_legacy_fh.close()

    return {
        "fps_used": fps_used,
        "frames_seen": frames_seen,
        "frames_inferred": frames_inferred,
        "wheels_detected_total": wheels_detected_total,
        "frame_index": frame_index_list,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    source = args.source
    source_type = resolve_source_type(source)
    if source_type == "unknown":
        raise SystemExit(
            f"--source must be a directory or a video file ({', '.join(VIDEO_EXTS)}); "
            f"got: {source}"
        )

    if args.every_n_frames < 1:
        raise SystemExit(f"--every-n-frames must be >= 1, got {args.every_n_frames}")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit(f"--max-frames must be >= 1, got {args.max_frames}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.name if source_type == "directory" else source.stem

    started_at = time.time()
    # Import YOLO lazily so --help / tests don't pay for it.
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "pose":
        print(
            f"WARNING: model task is {getattr(model, 'task', '?')!r}, expected 'pose'."
        )

    if source_type == "directory":
        stats = _run_directory(
            model, source=source, args=args, out_dir=args.out_dir, stem=stem
        )
    else:
        stats = _run_video(
            model, source=source, args=args, out_dir=args.out_dir, stem=stem
        )

    duration_seconds = time.time() - started_at

    # `frame_index` is an extension beyond the spec'd 14 keys: a per-frame
    # manifest so AR can load the summary and correlate frame_ids with their
    # JSON paths without scanning out_dir. Keep this if extended downstream.
    summary = {
        "source": str(source.resolve()),
        "source_type": source_type,
        "model": str(Path(args.model).resolve()),
        "device": args.device,
        "fps_used": stats["fps_used"],
        "frames_seen": stats["frames_seen"],
        "frames_inferred": stats["frames_inferred"],
        "frames_subsampled": args.every_n_frames,
        "wheels_detected_total": stats["wheels_detected_total"],
        "duration_seconds": duration_seconds,
        "started_at": started_at,
        "thresholds": {"conf": args.conf, "iou": args.iou, "max_det": args.max_det},
        "output_mode": "combined_jsonl" if args.combined_jsonl else "per_frame",
        "primary_schema": "confirmed",
        "legacy_companion_emitted": bool(args.emit_legacy),
        "frame_index": stats["frame_index"],
    }
    summary_path = args.out_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Source:               {source} ({source_type})")
    print(f"Model:                {args.model}")
    print(f"Device:               {args.device}")
    print(f"fps used:             {stats['fps_used']}")
    print(f"Frames seen:          {stats['frames_seen']}")
    print(f"Frames inferred:      {stats['frames_inferred']}")
    print(f"Subsample step:       {args.every_n_frames}")
    print(f"Wheels detected:      {stats['wheels_detected_total']}")
    print(f"Duration (s):         {duration_seconds:.2f}")
    print(f"Output mode:          {summary['output_mode']}")
    print(f"Primary schema:       {summary['primary_schema']}")
    print(f"Legacy companion:     {summary['legacy_companion_emitted']}")
    print(f"Summary JSON:         {summary_path}")


if __name__ == "__main__":
    main()

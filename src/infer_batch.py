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

Usage:
    python src/infer_batch.py \\
        --source data/wheel_dataset/images/val \\
        --model runs/pose/wheel_v3/weights/best.pt \\
        --out-dir /tmp/batch_out \\
        --device cpu --max-frames 3 --target-schema

Output layout (per-frame mode, default):
    <out-dir>/<stem>__frame_000000.json         AR legacy payload
    <out-dir>/<stem>__frame_000000_target.json  if --target-schema
    <out-dir>/batch_summary.json                totals + frame index

Combined-jsonl mode (--combined-jsonl):
    <out-dir>/<stem>.jsonl                      one payload per line
    <out-dir>/<stem>_target.jsonl               if --target-schema
    <out-dir>/batch_summary.json

Note: ``batch_summary.json`` includes a ``frame_index`` array (per-frame
manifest of frame_id + JSON path) as an extension beyond the spec's 14
keys, so AR can correlate frames to outputs without scanning out_dir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from postprocess_wheels import (
    N_KEYPOINTS,
    build_ar_payload,
    to_target_schema,
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
        vis = 2 if c >= 0.5 else (1 if c >= 0.15 else 0)
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
        "--target-schema",
        action="store_true",
        help="Additionally emit AR-target-schema JSON per frame.",
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


def _build_payloads(
    detections: list[dict],
    *,
    conf: float,
    frame_id: str,
    timestamp: float,
    img_size: list[int],
    thresholds: dict,
    want_target: bool,
) -> tuple[dict, dict | None]:
    """Build the AR legacy payload and (optionally) the target-schema one.

    Mirrors the shape produced by ``infer_image.py`` so AR sees the same
    contract regardless of which entry point we use.
    """
    ar_payload = build_ar_payload(
        detections, conf_threshold=conf, frame_id=frame_id, timestamp=timestamp
    )
    target_payload = to_target_schema(ar_payload) if want_target else None
    ar_payload["image_size"] = img_size
    ar_payload["thresholds"] = thresholds
    return ar_payload, target_payload


def _write_per_frame(
    out_dir: Path,
    stem: str,
    frame_index: int,
    ar_payload: dict,
    target_payload: dict | None,
) -> tuple[Path, Path | None]:
    """Persist one frame as ``<stem>__frame_<i:06d>.json`` (+ target)."""
    base = out_dir / f"{stem}__frame_{frame_index:06d}"
    json_path = base.with_suffix(".json")
    json_path.write_text(
        json.dumps(ar_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    target_path: Path | None = None
    if target_payload is not None:
        target_path = out_dir / f"{stem}__frame_{frame_index:06d}_target.json"
        target_path.write_text(
            json.dumps(target_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return json_path, target_path


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

    jsonl_fh = None
    jsonl_target_fh = None
    if args.combined_jsonl:
        jsonl_fh = (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8")
        if args.target_schema:
            jsonl_target_fh = (out_dir / f"{stem}_target.jsonl").open(
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

            ar_payload, target_payload = _build_payloads(
                detections,
                conf=args.conf,
                frame_id=frame_id,
                timestamp=timestamp,
                img_size=img_size,
                thresholds=thresholds,
                want_target=args.target_schema,
            )
            ar_payload["image"] = str(img_path)

            wheels_detected_total += len(ar_payload["wheels"])
            frames_inferred += 1

            if args.combined_jsonl:
                if jsonl_fh is None:
                    raise RuntimeError(
                        "jsonl_fh is None inside combined-jsonl branch — bug"
                    )
                jsonl_fh.write(json.dumps(ar_payload, ensure_ascii=False) + "\n")
                if jsonl_target_fh is not None and target_payload is not None:
                    jsonl_target_fh.write(
                        json.dumps(target_payload, ensure_ascii=False) + "\n"
                    )
                frame_index_list.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "source_image": str(img_path),
                        "n_wheels": len(ar_payload["wheels"]),
                    }
                )
            else:
                json_path, target_path = _write_per_frame(
                    out_dir, stem, original_idx, ar_payload, target_payload
                )
                frame_index_list.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "source_image": str(img_path),
                        "n_wheels": len(ar_payload["wheels"]),
                        "json": str(json_path),
                        "target_json": str(target_path) if target_path else None,
                    }
                )
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()
        if jsonl_target_fh is not None:
            jsonl_target_fh.close()

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

    jsonl_fh = None
    jsonl_target_fh = None
    if args.combined_jsonl:
        jsonl_fh = (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8")
        if args.target_schema:
            jsonl_target_fh = (out_dir / f"{stem}_target.jsonl").open(
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

                ar_payload, target_payload = _build_payloads(
                    detections,
                    conf=args.conf,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    img_size=img_size,
                    thresholds=thresholds,
                    want_target=args.target_schema,
                )
                ar_payload["image"] = f"{source}#frame={original_idx}"

                wheels_detected_total += len(ar_payload["wheels"])
                frames_inferred += 1

                if args.combined_jsonl:
                    if jsonl_fh is None:
                        raise RuntimeError(
                            "jsonl_fh is None inside combined-jsonl branch — bug"
                        )
                    jsonl_fh.write(json.dumps(ar_payload, ensure_ascii=False) + "\n")
                    if jsonl_target_fh is not None and target_payload is not None:
                        jsonl_target_fh.write(
                            json.dumps(target_payload, ensure_ascii=False) + "\n"
                        )
                    frame_index_list.append(
                        {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "original_frame_index": original_idx,
                            "n_wheels": len(ar_payload["wheels"]),
                        }
                    )
                else:
                    json_path, target_path = _write_per_frame(
                        out_dir, stem, original_idx, ar_payload, target_payload
                    )
                    frame_index_list.append(
                        {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "original_frame_index": original_idx,
                            "n_wheels": len(ar_payload["wheels"]),
                            "json": str(json_path),
                            "target_json": str(target_path) if target_path else None,
                        }
                    )
            finally:
                original_idx += 1
    finally:
        cap.release()
        if jsonl_fh is not None:
            jsonl_fh.close()
        if jsonl_target_fh is not None:
            jsonl_target_fh.close()

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
        "target_schema_emitted": bool(args.target_schema),
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
    print(f"Target schema:        {summary['target_schema_emitted']}")
    print(f"Summary JSON:         {summary_path}")


if __name__ == "__main__":
    main()

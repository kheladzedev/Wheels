"""Build a 3D-eval frames manifest by running the champion YOLO-pose model
on an existing MCP export's images, so the 3D gate can operate on real
model predictions rather than ground-truth 2D keypoints.

The script reuses EXACTLY the same camera-pose source as
``scripts/make_eval3d_manifest_from_ue_v0_2.py`` (full UE pose from rich
annotations, built via ``camera_from_ue_pose.camera_from_ue_pose``), so
geometry is comparable between the GT-2D manifest and this one.

KEY DIFFERENCES from the GT-2D builder:
  - ``points_source = "model_prediction"``   (not "ue_ground_truth")
  - ``source = "real"``                      (enables the gate to flip green
                                              once geometry is correct)
  - Per-wheel A/B/C come from YOLO inference, NOT export keypoints.
  - A frame is included only if the model detects exactly the right number
    of wheels for which GT is available, and all detected wheels pass the
    confirmed-schema visibility guard.

SCENE MATCHING — the hard part:
  The MCP export annotates multiple static ``WheelMarker`` actors per
  turntable frame.  A frame with N markers produces N rows in the GT
  annotation.  After model inference on the same frame we get up to N
  predicted wheels.  We match predicted wheels to GT actors by nearest
  bbox-center distance, then take the one best match per actor.
  Frames where the model does not detect any wheels, or where the match
  is ambiguous (two predictions closest to the same actor), are skipped.

LIMITATIONS (honest — see provenance string in emitted manifest):
  - ``ab_contract``: the floor-ray A/B contract requires that the model
    has been trained to place A/B on the floor-contact strip.  The
    current champion may still place A/B near the rim (the training
    label set mirrors the UE GT, which uses rim spheres for a/b).
    The gate will measure this: if disc-height error is high and sigma is
    low, the systematic A/B drift is confirmed.
  - ``gt_disc_position`` is emitted when the export provides the GT
    disc-center world point.  The harness still scores what the model
    *sends* in 2D; the GT point is used only as the height-error anchor
    that catches systematic A/B floor-ray drift which sigma alone can miss.

Usage::

    python scripts/make_eval3d_manifest_from_model_predictions.py \\
        --dataset-root /path/to/WheelsDataset_v0_2 \\
        --weights runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \\
        --out outputs/eval3d/eval3d_manifest_model_pred_v0_2.json

    Optional:
        --device mps|cpu|0        inference device (default: auto)
        --conf  0.25              wheel detection confidence threshold
        --min-frames 2            drop scenes with fewer accepted frames
        --max-frames-per-scene N  turntable cap per actor (default: no cap)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import camera_from_ue_pose as cp  # noqa: E402

# Keypoint names used by the CONFIRMED schema (output of to_confirmed_schema).
# These align with INTERNAL_TO_CONFIRMED_KP in postprocess_wheels.py:
#   rim_left  -> a
#   rim_right -> b
#   disc_bottom -> c_disc_bottom
CONFIRMED_POINT_KEYS = ("a", "b", "c_disc_bottom")

# Names of the GT keypoints in the rich annotation that play the role of
# scene-grouping; used ONLY to build scene metadata (not for 2D points).
A_NAME, B_NAME, C_NAME = "SphereLeft", "SphereRight", "Center"
REQUIRED_GT = (A_NAME, B_NAME, C_NAME)


# ---------------------------------------------------------------------------
# Pure helpers (no GPU, no cv2, fully testable)
# ---------------------------------------------------------------------------


def _visible_in_image(kp_image: dict, vis: dict, w: int, h: int) -> bool:
    """True iff all three GT keypoints are marked visible and within the image."""
    for name in REQUIRED_GT:
        if name not in kp_image or not vis.get(name, True):
            return False
        x, y = kp_image[name]
        if not (0.0 <= x < w and 0.0 <= y < h):
            return False
    return True


def _bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    """Return the (cx, cy) of an xyxy bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _gt_bbox_center_px(kp_image: dict, names: tuple) -> tuple[float, float] | None:
    """Rough GT bbox center from GT keypoint pixel coords.

    We approximate the wheel center as the centroid of the visible GT
    keypoints — good enough for nearest-neighbour matching.
    """
    pts = [kp_image[n] for n in names if n in kp_image]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def match_predictions_to_actors(
    pred_wheels: list[dict],
    gt_actors: list[dict],
    *,
    require_exact_count: bool = False,
    reject_duplicate_nearest: bool = False,
) -> dict[str, dict | None] | None:
    """Match each GT actor to its best predicted wheel by bbox-center distance.

    ``pred_wheels`` — list of confirmed-schema wheel dicts:
        {``bbox_xyxy``: [x1,y1,x2,y2], ``confidence``: float,
         ``points``: {``a``, ``b``, ``c_disc_bottom``}}
    ``gt_actors`` — list of dicts:
        {``actor``: str, ``kp_image_center``: (cx, cy)}

    Returns {actor_name: wheel_dict | None}.  A prediction is assigned to
    at most one actor (greedy nearest-neighbour); actors with no close
    prediction get None.  In strict mode, returns None for the whole frame
    when the prediction count differs from the GT actor count, or when two
    predictions are nearest to the same GT actor.

    Pure: no model, no file I/O.
    """
    if require_exact_count and len(pred_wheels) != len(gt_actors):
        return None

    if not pred_wheels or not gt_actors:
        return {a["actor"]: None for a in gt_actors}

    pred_centers = [_bbox_center(w["bbox_xyxy"]) for w in pred_wheels]
    if reject_duplicate_nearest:
        nearest_actors: list[str] = []
        for pc in pred_centers:
            nearest_actor = min(
                gt_actors,
                key=lambda a: (
                    (pc[0] - a["kp_image_center"][0]) ** 2
                    + (pc[1] - a["kp_image_center"][1]) ** 2
                )
                ** 0.5,
            )["actor"]
            nearest_actors.append(nearest_actor)
        if len(set(nearest_actors)) != len(nearest_actors):
            return None

    assigned_pred: set[int] = set()
    result: dict[str, dict | None] = {}

    for actor_info in gt_actors:
        cgt = actor_info["kp_image_center"]
        best_dist = float("inf")
        best_idx = -1
        for pi, pc in enumerate(pred_centers):
            if pi in assigned_pred:
                continue
            d = ((pc[0] - cgt[0]) ** 2 + (pc[1] - cgt[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = pi
        if best_idx >= 0:
            assigned_pred.add(best_idx)
            result[actor_info["actor"]] = pred_wheels[best_idx]
        else:
            result[actor_info["actor"]] = None

    return result


def extract_points_from_confirmed_wheel(wheel: dict) -> dict[str, list[float]] | None:
    """Pull {a, b, c_disc_bottom} from a confirmed-schema wheel dict.

    Returns None if any of the three keys is missing.
    """
    pts = wheel.get("points", {})
    if not all(k in pts for k in CONFIRMED_POINT_KEYS):
        return None
    return {k: list(pts[k]) for k in CONFIRMED_POINT_KEYS}


def build_scene_table(
    ann_dir: Path,
    *,
    max_frames_per_scene: int | None = None,
    min_frames: int = 2,
) -> tuple[dict[str, dict], int, int]:
    """Read the rich annotation directory and build the bare scene table.

    Returns
        scenes        — {actor: {"pose_frames": [...], "_gt_world": list|None}}
        frames_seen   — total annotation files processed
        frames_dropped — annotation files with no usable GT wheel

    Each entry in ``pose_frames`` is:
        {"frame_id": str, "image_size": [W,H], "pose": {...},
         "image_path": Path,
         "gt_actors": [{"actor": str, "kp_image_center": (cx,cy)}]}

    The model-prediction side populates ``points`` later; this is the
    pure pose-harvest step.
    """
    scenes: dict[str, dict] = {}
    frames_seen = 0
    frames_dropped = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        frames_seen += 1
        d = json.loads(ann_path.read_text(encoding="utf-8"))
        w, h = int(d["image_width"]), int(d["image_height"])
        cam = d["camera"]
        pose = {
            "location": cam["location"],
            "rotation": cam["rotation"],
            "fov": cam["fov"],
        }

        # Find the image file that matches this annotation.
        # The MCP export uses the same stem, living in an ``images/`` sibling.
        img_stem = ann_path.stem
        img_dir = ann_dir.parent / "images"
        img_path: Path | None = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = img_dir / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        gt_actors_this_frame: list[dict] = []
        frame_useful = False
        for wheel in d["wheels"]:
            ki = wheel["keypoints_image"]
            vis = wheel.get("visibility", {})
            kw = wheel.get("keypoints_world", {})
            if not _visible_in_image(ki, vis, w, h):
                continue
            actor = wheel.get("actor") or f"stencil_{wheel.get('stencil_id')}"
            scene = scenes.setdefault(
                actor,
                {"pose_frames": [], "_gt_world": None, "img_w": w, "img_h": h},
            )
            if (
                max_frames_per_scene is not None
                and len(scene["pose_frames"]) >= max_frames_per_scene
            ):
                continue
            ctr = _gt_bbox_center_px(ki, REQUIRED_GT)
            gt_actors_this_frame.append({"actor": actor, "kp_image_center": ctr})
            scene["pose_frames"].append(
                {
                    "frame_id": f"{actor}__{d.get('frame', ann_path.stem)}",
                    "image_size": [w, h],
                    "pose": pose,
                    "image_path": img_path,
                    "gt_actors_in_frame": gt_actors_this_frame,
                }
            )
            if scene["_gt_world"] is None and C_NAME in kw:
                scene["_gt_world"] = cp.ue_world_to_rh(kw[C_NAME]).tolist()
            frame_useful = True

        if not frame_useful:
            frames_dropped += 1

    return scenes, frames_seen, frames_dropped


def assemble_manifest(
    scenes: dict[str, dict],
    *,
    dataset_name: str,
    weights_path: str,
    min_frames: int = 2,
) -> dict:
    """Assemble the final manifest dict from completed scene dicts.

    Each scene dict must have ``frames`` (already populated with ``points``),
    and optionally ``_gt_world``.

    Pure: called after inference is done, no GPU/file I/O here.
    """
    out_scenes: dict[str, dict] = {}
    for actor, scene in sorted(scenes.items()):
        frames = scene.get("frames", [])
        if len(frames) < min_frames:
            continue
        sc: dict = {"frames": frames}
        if scene.get("_gt_world") is not None:
            sc["gt_disc_position"] = scene["_gt_world"]
        out_scenes[actor] = sc

    img_w = next((s["img_w"] for s in scenes.values() if "img_w" in s), 1280)
    img_h = next((s["img_h"] for s in scenes.values() if "img_h" in s), 720)

    return {
        "units": "cm",
        "image_size": [img_w, img_h],
        "source": "real",
        "geometry_source": f"real_ue_{dataset_name}",
        "points_source": "model_prediction",
        "ab_contract": "model_predicted_floor_ray",
        "provenance": (
            f"Real MCP export {dataset_name}: parity-certified camera pose "
            f"(camera_from_ue_pose) + 2D points from champion model "
            f"({weights_path}). points_source='model_prediction' — this "
            "manifest is a REAL gate candidate. Whether the gate is green "
            "depends on whether the model has learned floor-ray A/B (not "
            "rim spheres); high disc-height error with low sigma indicates "
            "the A/B systematic drift is still present."
        ),
        "scenes": out_scenes,
    }


# ---------------------------------------------------------------------------
# Inference driver (GPU-touching — only called from main())
# ---------------------------------------------------------------------------


def _run_inference_on_scenes(
    scenes: dict[str, dict],
    *,
    weights: Path,
    device: str | None,
    conf: float,
    iou: float,
) -> dict[str, dict]:
    """Run YOLO on every unique image in ``scenes``, populate ``frames``.

    Mutates ``scenes`` in place: each pose_frame dict gets a ``points``
    key when inference succeeds.  Returns a new dict keyed by actor with
    only the accepted frames under key ``frames``.
    """
    from ultralytics import YOLO  # noqa: PLC0415 — lazy import
    from postprocess_wheels import build_ar_payload, to_confirmed_schema  # noqa: PLC0415

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "pose":
        print(f"WARNING: model task={getattr(model, 'task', '?')!r}, expected 'pose'.")

    # Build a map: image_path -> list of pose_frame records that use it.
    # Multiple actors can share the same image (turntable captures).
    img_to_frames: dict[Path | None, list[tuple[str, dict]]] = {}
    for actor, scene in scenes.items():
        for pf in scene["pose_frames"]:
            img_path = pf["image_path"]
            img_to_frames.setdefault(img_path, []).append((actor, pf))

    # Accumulate accepted per-actor frames.
    actor_frames: dict[str, list[dict]] = {}

    for img_path, frame_records in sorted(
        img_to_frames.items(), key=lambda kv: str(kv[0])
    ):
        if img_path is None:
            # No image found for this annotation frame — skip.
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        result = results[0]

        # Build confirmed-schema wheels from raw result.
        from infer_batch import detections_from_result  # noqa: PLC0415

        detections = detections_from_result(result, conf, max_det=20)
        legacy = build_ar_payload(detections, conf_threshold=conf)
        confirmed = to_confirmed_schema(legacy)
        pred_wheels = confirmed.get("wheels", [])

        # Collect the GT-actor info for every actor visible in this frame.
        # We reconstruct it from the frame records (one entry per actor).
        gt_actors_in_frame: list[dict] = []
        for actor, pf in frame_records:
            # Each pose_frame has ``gt_actors_in_frame`` with all actors
            # visible in that raw annotation frame, not just the current actor.
            # Grab the set from the first record (they share the same frame).
            gt_actors_in_frame = pf.get("gt_actors_in_frame", [])
            break

        matches = match_predictions_to_actors(
            pred_wheels,
            gt_actors_in_frame,
            require_exact_count=True,
            reject_duplicate_nearest=True,
        )
        if matches is None:
            continue

        for actor, pf in frame_records:
            matched_wheel = matches.get(actor)
            if matched_wheel is None:
                continue
            pts = extract_points_from_confirmed_wheel(matched_wheel)
            if pts is None:
                continue

            frame_entry = {
                "frame_id": pf["frame_id"],
                "image_size": pf["image_size"],
                "pose": pf["pose"],
                "points": pts,
            }
            actor_frames.setdefault(actor, []).append(frame_entry)

    # Merge accepted frames back into scene dicts.
    result_scenes: dict[str, dict] = {}
    for actor, scene in scenes.items():
        accepted = actor_frames.get(actor, [])
        result_scenes[actor] = dict(scene)
        result_scenes[actor]["frames"] = accepted

    return result_scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument(
        "--weights",
        type=Path,
        default=Path(DEFAULT_WEIGHTS),
        help="Champion YOLO-pose weights (.pt).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval3d/eval3d_manifest_model_pred_v0_2.json"),
    )
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--min-frames", type=int, default=2)
    p.add_argument("--max-frames-per-scene", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ann_dir = args.dataset_root / "annotations"
    if not ann_dir.is_dir():
        print(
            f"ERROR: {ann_dir} not found.  "
            "Pass --dataset-root pointing at WheelsDataset_v0_2/."
        )
        return 1

    print(f"[model-pred-manifest] Harvesting pose from {ann_dir} ...")
    scenes, frames_seen, frames_dropped = build_scene_table(
        ann_dir,
        max_frames_per_scene=args.max_frames_per_scene,
        min_frames=args.min_frames,
    )
    print(
        f"[model-pred-manifest] annotations={frames_seen} "
        f"dropped={frames_dropped} actors={len(scenes)}"
    )

    print(f"[model-pred-manifest] Running inference ({args.weights}) ...")
    populated_scenes = _run_inference_on_scenes(
        scenes,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    manifest = assemble_manifest(
        populated_scenes,
        dataset_name=args.dataset_root.name,
        weights_path=str(args.weights),
        min_frames=args.min_frames,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_scenes = len(manifest["scenes"])
    n_frames = sum(len(s["frames"]) for s in manifest["scenes"].values())
    print(
        f"[model-pred-manifest] scenes={n_scenes} frames={n_frames} "
        f"points_source={manifest['points_source']!r} -> {args.out}"
    )
    if n_scenes == 0:
        print(
            "[model-pred-manifest] WARNING: zero scenes emitted. "
            "Check that the images/ directory exists next to annotations/ "
            "and that the model detects wheels in those images."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

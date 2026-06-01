"""Export a trained YOLO-pose checkpoint to ONNX / CoreML / TFLite + sanity-check.

Stage 6 prep: hand the AR / web / iOS / Android teams a one-shot way to
convert a `.pt` checkpoint to their runtime format and verify the
exported model still predicts the same wheels and keypoints as the
PyTorch original.

What the script does:

  1. Calls ``ultralytics.YOLO(model_path).export(format=...)`` with the
     knobs Ultralytics exposes (``imgsz``, ``device``, ``half``, ``int8``,
     ``simplify``, ``dynamic``). Defaults match the Ultralytics defaults
     so AR can pick the lightest export the format supports.
  2. Reloads both the original ``.pt`` and the exported file via
     ``YOLO(path)``, runs one prediction on a sample image, and
     compares bbox / keypoint / confidence with loose tolerances
     (2 px bbox, 3 px keypoint, 0.05 confidence) — quantized formats
     drift a few px and we want that to be OK. If anything outside
     tolerance, exit non-zero.

The pure helpers (``compare_detections``, ``pick_sample_image``,
``infer_one``) are deliberately importable and live without
ultralytics, so tests can exercise the matching logic without
instantiating a model.

Final export format pending Q10 in docs/QUESTIONS_FOR_TEAM.md — once
the AR team picks the production runtime, this script doesn't change,
the CLI flag does.

Usage:
    python src/export_model.py --model runs/pose/wheel_v3/weights/best.pt \\
        --format onnx --device cpu
    python src/export_model.py --model runs/pose/wheel_v3/weights/best.pt \\
        --format coreml --imgsz 640
    python src/export_model.py --model runs/pose/wheel_v3/weights/best.pt \\
        --format mlmodel --imgsz 640 --no-sanity
    python src/export_model.py --model runs/pose/wheel_v3/weights/best.pt \\
        --format tflite --int8

TFLite needs ``tensorflow`` in the venv — Ultralytics will surface its own
import error if missing. Install it yourself (we do not auto-install).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    # Ultralytics is imported lazily in main() so the pure helpers stay
    # usable without it. Type-checkers still get the proper annotation.
    from ultralytics import YOLO

# Tolerances picked to swallow quantization drift (a few px is normal for
# TFLite int8 / CoreML fp16) while still catching genuine breakage like a
# missed detection or a swapped keypoint.
DEFAULT_BBOX_ATOL = 2.0
DEFAULT_KP_ATOL = 3.0
DEFAULT_CONF_ATOL = 0.05

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 20

VALID_FORMATS = ("onnx", "coreml", "mlmodel", "tflite")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DATASET_VAL_DIR = Path("data/wheel_dataset/images/val")


# ---------------------------------------------------------------------------
# Pure helpers — importable, no ultralytics dependency.
# ---------------------------------------------------------------------------


def _box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """IoU of two axis-aligned boxes in xyxy pixel coordinates.

    Copied (not imported) from ``src/eval_keypoints.py`` on purpose: the
    export script must keep working even if the eval module shifts shape
    later. The two scripts are independent by design.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _greedy_match_detections(
    pt_dets: Sequence[dict],
    ex_dets: Sequence[dict],
) -> list[tuple[int, int]]:
    """Greedy IoU match: PT detections (sorted by conf desc) take the
    highest-IoU still-unmatched exported detection.

    Returns a list of ``(pt_idx, exported_idx)`` pairs. Unmatched PT
    detections are not represented — the caller checks for them via the
    overall count diff. We don't gate on a minimum IoU because the
    comparison stage already enforces a 2 px bbox tolerance: if two
    detections are within 2 px in xyxy, their IoU is necessarily high.
    Using a hard IoU floor here would let count mismatches sneak past
    when bboxes happen to align but confidence drifted (and the eval is
    supposed to catch that).
    """
    order = sorted(
        range(len(pt_dets)),
        key=lambda i: pt_dets[i].get("conf", 0.0),
        reverse=True,
    )
    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for p_idx in order:
        best_e = -1
        best_iou = -1.0
        for e_idx in range(len(ex_dets)):
            if e_idx in taken:
                continue
            iou = _box_iou(pt_dets[p_idx]["bbox"], ex_dets[e_idx]["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_e = e_idx
        if best_e >= 0:
            pairs.append((p_idx, best_e))
            taken.add(best_e)
    return pairs


def compare_detections(
    pt_result: dict,
    exported_result: dict,
    *,
    bbox_atol: float = DEFAULT_BBOX_ATOL,
    kp_atol: float = DEFAULT_KP_ATOL,
    conf_atol: float = DEFAULT_CONF_ATOL,
) -> dict:
    """Compare PT vs exported detection sets at loose pixel tolerances.

    Inputs share the shape ``{"detections": [{"bbox": [x1,y1,x2,y2],
    "conf": float, "keypoints": [[x,y], ...]}, ...]}``.

    Returns a report:
        {
            "matched": bool,
            "n_pt": int,
            "n_exported": int,
            "max_bbox_drift_px": float,   # L-inf over xyxy coordinates
            "max_kp_drift_px": float,     # max single coord drift
            "max_conf_drift": float,
            "failures": list[str],        # human-readable reasons
        }

    A count mismatch (e.g. PT finds 4 wheels, exported finds 3) is itself
    a failure — quantization shouldn't drop or invent detections at the
    same conf threshold.
    """
    pt_dets = pt_result.get("detections", [])
    ex_dets = exported_result.get("detections", [])

    failures: list[str] = []
    max_bbox = 0.0
    max_kp = 0.0
    max_conf = 0.0
    pair_diagnostics: list[dict] = []

    if len(pt_dets) != len(ex_dets):
        failures.append(
            f"detection count differs: pt={len(pt_dets)} exported={len(ex_dets)}"
        )

    pairs = _greedy_match_detections(pt_dets, ex_dets)

    for pt_idx, ex_idx in pairs:
        pt = pt_dets[pt_idx]
        ex = ex_dets[ex_idx]
        pair_report: dict = {
            "pt_idx": pt_idx,
            "exported_idx": ex_idx,
            "iou": _box_iou(pt["bbox"], ex["bbox"]),
            "coordinate_scale_warning": False,
        }

        # bbox: per-coordinate abs drift, take the max.
        pt_bbox = np.asarray(pt["bbox"], dtype=np.float64)
        ex_bbox = np.asarray(ex["bbox"], dtype=np.float64)
        bbox_drift = float(np.max(np.abs(pt_bbox - ex_bbox))) if pt_bbox.size else 0.0
        pair_report["bbox_drift_px"] = bbox_drift
        if pt_bbox.size and ex_bbox.size:
            pair_report["pt_bbox_max_coord"] = float(np.max(np.abs(pt_bbox)))
            pair_report["exported_bbox_max_coord"] = float(np.max(np.abs(ex_bbox)))
            if pair_report["pt_bbox_max_coord"] > 32.0 and pair_report["exported_bbox_max_coord"] <= 2.0:
                pair_report["coordinate_scale_warning"] = True
        max_bbox = max(max_bbox, bbox_drift)
        if not np.allclose(pt_bbox, ex_bbox, atol=bbox_atol):
            failures.append(
                f"bbox drift exceeds {bbox_atol}px at pt_idx={pt_idx} "
                f"exported_idx={ex_idx}: pt={pt_bbox.tolist()} "
                f"exported={ex_bbox.tolist()} (max coord drift {bbox_drift:.3f}px)"
            )

        # confidence: scalar drift.
        pt_conf = float(pt.get("conf", 0.0))
        ex_conf = float(ex.get("conf", 0.0))
        conf_drift = abs(pt_conf - ex_conf)
        pair_report["pt_conf"] = pt_conf
        pair_report["exported_conf"] = ex_conf
        pair_report["conf_drift"] = conf_drift
        max_conf = max(max_conf, conf_drift)
        if conf_drift > conf_atol:
            failures.append(
                f"conf drift exceeds {conf_atol} at pt_idx={pt_idx} "
                f"exported_idx={ex_idx}: pt={pt_conf:.3f} "
                f"exported={ex_conf:.3f} (drift {conf_drift:.3f})"
            )

        # keypoints: per-coordinate abs drift across all kps.
        pt_kp = np.asarray(pt.get("keypoints", []), dtype=np.float64)
        ex_kp = np.asarray(ex.get("keypoints", []), dtype=np.float64)
        if pt_kp.shape != ex_kp.shape:
            pair_report["keypoint_shape"] = {
                "pt": list(pt_kp.shape),
                "exported": list(ex_kp.shape),
            }
            failures.append(
                f"keypoint shape differs at pt_idx={pt_idx} "
                f"exported_idx={ex_idx}: pt={pt_kp.shape} exported={ex_kp.shape}"
            )
        elif pt_kp.size:
            kp_drift = float(np.max(np.abs(pt_kp - ex_kp)))
            pair_report["keypoint_drift_px"] = kp_drift
            max_kp = max(max_kp, kp_drift)
            if not np.allclose(pt_kp, ex_kp, atol=kp_atol):
                failures.append(
                    f"keypoint drift exceeds {kp_atol}px at pt_idx={pt_idx} "
                    f"exported_idx={ex_idx} (max drift {kp_drift:.3f}px)"
                )
        else:
            pair_report["keypoint_drift_px"] = 0.0
        pair_diagnostics.append(pair_report)

    matched = not failures
    return {
        "matched": matched,
        "n_pt": len(pt_dets),
        "n_exported": len(ex_dets),
        "max_bbox_drift_px": max_bbox,
        "max_kp_drift_px": max_kp,
        "max_conf_drift": max_conf,
        "pair_diagnostics": pair_diagnostics,
        "failures": failures,
    }


def pick_sample_image(
    arg_path: Path | None,
    dataset_root: Path = DATASET_VAL_DIR,
) -> Path:
    """Resolve the image to use for the sanity check.

    Honours an explicit ``--sample-image`` arg if it exists. Otherwise
    falls back to the first image in the dataset's val split (sorted
    lexicographically so the choice is deterministic across runs). If
    neither is available, raise ``FileNotFoundError`` — the sanity
    check needs at least one image and silently skipping it would hide
    quantization bugs.
    """
    if arg_path is not None:
        if arg_path.exists():
            return arg_path
        raise FileNotFoundError(f"--sample-image points at a missing file: {arg_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"No --sample-image given and dataset val dir does not exist: "
            f"{dataset_root}. Pass --sample-image explicitly or "
            "create data/wheel_dataset/images/val/."
        )
    candidates = sorted(
        p
        for p in dataset_root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not candidates:
        raise FileNotFoundError(
            f"No --sample-image given and {dataset_root} contains no "
            f"images (extensions checked: {IMAGE_EXTS}). Either populate "
            "the val split or pass --sample-image."
        )
    return candidates[0]


def infer_one(
    model: "YOLO",
    image_path: Path,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    device: str | None = None,
) -> dict:
    """Run a single-image prediction and pack the first result into the
    comparison-report shape.

    Output:
        {
            "detections": [
                {"bbox": [x1,y1,x2,y2], "conf": float,
                 "keypoints": [[x,y], ...]},
                ...
            ]
        }

    Detections without keypoints (e.g. a plain detect model) get an empty
    list for ``keypoints`` rather than a missing key — keeps the shape
    contract uniform for the comparator.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )
    if not results:
        return {"detections": []}
    result = results[0]
    detections: list[dict] = []
    if result.boxes is None:
        return {"detections": []}
    n = len(result.boxes)
    for i in range(n):
        box = result.boxes[i]
        bbox = [float(v) for v in box.xyxy[0].tolist()]
        det_conf = float(box.conf.item())
        kps_xy: list[list[float]] = []
        if result.keypoints is not None:
            xy = result.keypoints.xy[i].cpu().numpy()
            kps_xy = [[float(xy[k, 0]), float(xy[k, 1])] for k in range(xy.shape[0])]
        detections.append({"bbox": bbox, "conf": det_conf, "keypoints": kps_xy})
    return {"detections": detections}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(
    report: dict,
    *,
    bbox_atol: float,
    kp_atol: float,
    conf_atol: float,
) -> str:
    """Pretty-print the comparison report for stdout.

    Keeps the line shape stable so CI scrapers can grep ``matched=``.
    """
    lines = [
        "Sanity check (pt vs exported):",
        f"  matched:            {report['matched']}",
        f"  detections (pt):    {report['n_pt']}",
        f"  detections (exp):   {report['n_exported']}",
        f"  max bbox drift:     {report['max_bbox_drift_px']:.3f} px (atol {bbox_atol})",
        f"  max keypoint drift: {report['max_kp_drift_px']:.3f} px (atol {kp_atol})",
        f"  max conf drift:     {report['max_conf_drift']:.3f} (atol {conf_atol})",
    ]
    if report["failures"]:
        lines.append("  failures:")
        for f in report["failures"]:
            lines.append(f"    - {f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a YOLO-pose checkpoint to ONNX/CoreML/TFLite and verify "
            "the exported model matches the PyTorch original on one image."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the trained YOLO-pose .pt checkpoint.",
    )
    p.add_argument(
        "--format",
        required=True,
        choices=VALID_FORMATS,
        help=(
            "Target export format. Final production format pending Q10 in "
            "docs/QUESTIONS_FOR_TEAM.md. Use 'mlmodel' for legacy CoreML "
            "neuralnetwork export when Python/coremltools cannot write "
            "ML Program .mlpackage blobs."
        ),
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference image size (default {DEFAULT_IMGSZ}).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device passed to Ultralytics (e.g. 'cpu', 'mps', '0').",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Export in half precision (fp16). Format-dependent.",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="Export in int8 quantization. TFLite-only in practice.",
    )
    p.add_argument(
        "--simplify",
        action="store_true",
        help="ONNX: run the model through onnxsim before saving.",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="ONNX: allow dynamic batch/spatial dimensions in the exported graph.",
    )
    p.add_argument(
        "--sample-image",
        type=Path,
        default=None,
        help=(
            "Image used for the post-export sanity check. Defaults to the "
            "first lexicographic entry in data/wheel_dataset/images/val/."
        ),
    )
    p.add_argument(
        "--no-sanity",
        action="store_true",
        help="Skip the numerical comparison after export.",
    )
    p.add_argument(
        "--exported-task",
        choices=("auto", "detect", "segment", "classify", "pose", "obb"),
        default="auto",
        help=(
            "Task hint when reloading the exported artifact for sanity check. "
            "Use 'pose' for backends such as TFLite when Ultralytics cannot "
            "infer the task from the exported file."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to move the exported file. Defaults to the .pt's directory.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Detection conf threshold for sanity check (default {DEFAULT_CONF}).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"NMS IoU threshold for sanity check (default {DEFAULT_IOU}).",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_DET,
        help=f"Max detections kept per image (default {DEFAULT_MAX_DET}).",
    )
    p.add_argument(
        "--bbox-atol",
        type=float,
        default=DEFAULT_BBOX_ATOL,
        help=f"Bbox xyxy tolerance in px (default {DEFAULT_BBOX_ATOL}).",
    )
    p.add_argument(
        "--kp-atol",
        type=float,
        default=DEFAULT_KP_ATOL,
        help=f"Keypoint xy tolerance in px (default {DEFAULT_KP_ATOL}).",
    )
    p.add_argument(
        "--conf-atol",
        type=float,
        default=DEFAULT_CONF_ATOL,
        help=f"Detection conf tolerance (default {DEFAULT_CONF_ATOL}).",
    )
    return p.parse_args()


def _build_export_kwargs(args: argparse.Namespace) -> dict:
    """Pick which export knobs to pass through to Ultralytics.

    Only flags the user explicitly set are forwarded — Ultralytics has
    its own per-format defaults and we don't want to override them
    accidentally. ``imgsz`` is the one knob we always send because
    callers expect the AR runtime input size to be reproducible.
    """
    kw: dict = {"format": args.format, "imgsz": args.imgsz}
    if args.device is not None:
        kw["device"] = args.device
    if args.half:
        kw["half"] = True
    if args.int8:
        kw["int8"] = True
    if args.simplify:
        kw["simplify"] = True
    if args.dynamic:
        kw["dynamic"] = True
    return kw


def _move_export(exported_path: Path, out_dir: Path) -> Path:
    """Relocate the Ultralytics-exported file under ``out_dir`` if needed.

    Ultralytics writes the exported file next to the source .pt. If the
    user gave a custom ``--out-dir``, move it there so downstream
    consumers (CI artifacts, mobile-side packaging) can pick it up.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / exported_path.name
    if exported_path.resolve() == target.resolve():
        return exported_path
    exported_path.replace(target)
    return target


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.model.is_file():
        raise FileNotFoundError(f"--model must be a file, got: {args.model}")

    # Import ultralytics lazily so `--help` doesn't pay the import cost
    # and the pure helpers stay importable in environments without it.
    from ultralytics import YOLO  # noqa: PLC0415

    print(f"Loading model: {args.model}")
    pt_model = YOLO(str(args.model))
    if getattr(pt_model, "task", None) != "pose":
        print(
            f"WARNING: model task is {getattr(pt_model, 'task', '?')!r}, "
            "expected 'pose'. Exporting anyway; sanity check may produce "
            "empty keypoints."
        )

    export_kwargs = _build_export_kwargs(args)
    print(f"Exporting with kwargs: {export_kwargs}")
    exported_path_str = pt_model.export(**export_kwargs)
    if exported_path_str is None:
        # Some backends (TFLite especially) hit a runtime error inside
        # Ultralytics' exporter and return None without raising. Path(None)
        # would then blow up with a useless TypeError; raise a clearer one.
        raise RuntimeError(
            "Ultralytics export() returned None — backend likely failed silently. "
            "Check the exporter logs above (e.g. missing tensorflow for tflite)."
        )
    exported_path = Path(exported_path_str)
    if not exported_path.exists():
        # Ultralytics returns a path string but doesn't always raise on
        # backend failures (TFLite is the usual culprit). Treat a
        # missing file as a hard error.
        raise RuntimeError(
            f"Ultralytics reported export to {exported_path} but the file "
            "does not exist. Check the exporter logs above for the "
            "underlying error (e.g. missing tensorflow for tflite)."
        )

    if args.out_dir is not None:
        exported_path = _move_export(exported_path, args.out_dir)
    print(f"Exported file: {exported_path}")

    if args.no_sanity:
        print("Sanity check skipped (--no-sanity).")
        return 0

    sample = pick_sample_image(args.sample_image)
    print(f"Sanity-check sample: {sample}")

    # Reload the PT model fresh so the export step's internal state
    # (the Exporter mutates a few attributes during conversion) doesn't
    # leak into the comparison.
    pt_model = YOLO(str(args.model))
    exported_model_kwargs = {}
    if args.exported_task != "auto":
        exported_model_kwargs["task"] = args.exported_task
    exported_model = YOLO(str(exported_path), **exported_model_kwargs)

    pt_result = infer_one(
        pt_model,
        sample,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )
    ex_result = infer_one(
        exported_model,
        sample,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )

    report = compare_detections(
        pt_result,
        ex_result,
        bbox_atol=args.bbox_atol,
        kp_atol=args.kp_atol,
        conf_atol=args.conf_atol,
    )
    print(
        format_report(
            report,
            bbox_atol=args.bbox_atol,
            kp_atol=args.kp_atol,
            conf_atol=args.conf_atol,
        )
    )

    return 0 if report["matched"] else 1


if __name__ == "__main__":
    sys.exit(main())

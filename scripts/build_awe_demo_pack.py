"""Build a controlled AWE-style demo pack.

Reuses already-rendered model-prediction overlays and the
confirmed-schema JSON sidecars under ``outputs/demo/`` (plus
auto-annotation drafts and synthetic-smoke previews) and adds a mock
AR visualisation per curated frame. The mock AR plane is drawn from
the JSON, not raycast — it is a visual explanation only.

Nothing in this script performs inference, training, schema changes,
or deletion. It writes only into ``outputs/awe_demo/`` (or the
``--out-dir`` argument override).

Usage:
    ./.venv/bin/python scripts/build_awe_demo_pack.py \
        --out-dir outputs/awe_demo \
        --count 15

Plan: see ``docs/AWE_DEMO_PLAN.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "outputs" / "awe_demo"
DEFAULT_PRED_OVERLAY_DIR = ROOT / "outputs" / "demo"
DEFAULT_PRED_JSON_DIR = ROOT / "outputs" / "demo" / "json"
DEFAULT_SOURCE_IMAGE_DIR = ROOT / "data" / "manual_real" / "images"
DEFAULT_ANNOTATION_PREVIEW_DIR = ROOT / "outputs" / "manual_real_auto_v3"
DEFAULT_SYNTHETIC_KEYPOINT_DIR = ROOT / "outputs" / "keypoint_preview"
DEFAULT_SYNTHETIC_POSE_DIR = ROOT / "outputs" / "pose_label_preview" / "train"

PROVENANCE_LABELS = (
    "model_prediction",
    "ar_mock_visualization",
    "annotation_preview",
    "synthetic_smoke",
    "debug_only",
)

BADGE_COLORS = {
    "model_prediction": (32, 120, 32),
    "ar_mock_visualization": (160, 80, 0),
    "annotation_preview": (40, 60, 200),
    "synthetic_smoke": (80, 80, 80),
    "debug_only": (16, 16, 200),
}

BADGE_TEXT = {
    "model_prediction": "MODEL PREDICTION (baseline demo — not production)",
    "ar_mock_visualization": "AR MOCK — visual explanation, not real raycast",
    "annotation_preview": "ANNOTATION DRAFT — not human-verified ground truth",
    "synthetic_smoke": "SYNTHETIC SMOKE — plumbing only, not real-world signal",
    "debug_only": "DEBUG ONLY",
}


@dataclass
class DemoItem:
    """One artefact in the demo pack."""

    stem: str
    provenance: str
    overlay_relpath: str
    source_relpath: str | None
    json_relpath: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stem": self.stem,
            "provenance": self.provenance,
            "overlay": self.overlay_relpath,
            "source": self.source_relpath,
            "json": self.json_relpath,
            "notes": list(self.notes),
        }


def annotate_with_badge(image: np.ndarray, provenance: str) -> np.ndarray:
    """Draw a top-left provenance badge onto a BGR image (in place)."""
    if provenance not in BADGE_COLORS:
        raise ValueError(f"unknown provenance: {provenance!r}")
    text = BADGE_TEXT[provenance]
    color = BADGE_COLORS[provenance]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(image.shape[1] / 1400.0, 1.1))
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = int(8 * scale)
    x0, y0 = pad, pad
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad + baseline
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x0 + pad, y0 + th + pad),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def _draw_point(
    image: np.ndarray, xy: Sequence[float], label: str, color: tuple[int, int, int]
) -> None:
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.circle(image, (x, y), 6, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), 9, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x + 10, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (x + 10, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_ar_mock_overlay(image_bgr: np.ndarray, wheels: Sequence[dict]) -> np.ndarray:
    """Draw a mock AR visualisation onto a copy of the source frame.

    For each wheel: bbox, A, B, C, the A-B base line (mock floor line),
    and a semi-transparent vertical rectangle above the base line that
    represents the recovered wheel plane. Strictly a *visual
    explanation*; no raycast or RANSAC happens here.
    """
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    h_img, w_img = canvas.shape[:2]
    for wheel in wheels:
        bbox = wheel.get("bbox_xyxy")
        points = wheel.get("points", {})
        if not bbox or not points:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_h = max(1.0, y2 - y1)
        cv2.rectangle(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (60, 200, 60),
            2,
            cv2.LINE_AA,
        )
        a = points.get("a")
        b = points.get("b")
        c = points.get("c_disc_bottom")
        if a is None or b is None or c is None:
            continue
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        plane_height = bbox_h * 1.1
        top_left = (int(round(ax)), int(round(ay - plane_height)))
        top_right = (int(round(bx)), int(round(by - plane_height)))
        bottom_left = (int(round(ax)), int(round(ay)))
        bottom_right = (int(round(bx)), int(round(by)))
        polygon = np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype=np.int32
        )
        cv2.fillPoly(overlay, [polygon], (200, 180, 60))
        cv2.line(canvas, bottom_left, bottom_right, (60, 220, 220), 3, cv2.LINE_AA)
        _draw_point(canvas, a, "A (floor)", (0, 200, 255))
        _draw_point(canvas, b, "B (floor)", (0, 200, 255))
        _draw_point(canvas, c, "C disc_bottom", (40, 80, 255))
    canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0.0)
    annotate_with_badge(canvas, "ar_mock_visualization")
    cv2.putText(
        canvas,
        "bbox + A/B (floor ray) + C (disc bottom)  |  vertical plane is drawn, not raycast",
        (10, h_img - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "bbox + A/B (floor ray) + C (disc bottom)  |  vertical plane is drawn, not raycast",
        (10, h_img - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _resolve_source_image(stem: str, source_dir: Path) -> Path | None:
    """Match a JSON / pred-overlay stem back to its source image."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = source_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def iter_model_prediction_frames(
    pred_overlay_dir: Path,
    pred_json_dir: Path,
    source_image_dir: Path,
    limit: int,
) -> list[tuple[Path, Path, Path]]:
    """Return (pred_overlay, json, source_image) triples, deterministically ordered."""
    if not pred_overlay_dir.is_dir() or not pred_json_dir.is_dir():
        return []
    overlays = sorted(pred_overlay_dir.glob("*_pred.jpg"))
    triples: list[tuple[Path, Path, Path]] = []
    for overlay in overlays:
        stem = overlay.stem.removesuffix("_pred")
        json_path = pred_json_dir / f"{stem}.json"
        if not json_path.is_file():
            continue
        source_image = _resolve_source_image(stem, source_image_dir)
        if source_image is None:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not payload.get("wheels"):
            continue
        triples.append((overlay, json_path, source_image))
        if len(triples) >= limit:
            break
    return triples


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _safe_relpath(path: Path, anchor: Path) -> str:
    """Return ``path`` relative to ``anchor`` when possible, else as-is."""
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return str(path)


def _badge_copy(src: Path, dst: Path, provenance: str) -> None:
    """Copy an image and burn a provenance badge onto the destination."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image {src}")
    annotate_with_badge(image, provenance)
    cv2.imwrite(str(dst), image)


def build_contact_sheet(
    items: Sequence[DemoItem],
    out_root: Path,
    columns: int = 4,
    tile_size: tuple[int, int] = (480, 320),
) -> Path | None:
    """Stitch overlay JPEGs into a single contact sheet image."""
    if not items:
        return None
    tile_w, tile_h = tile_size
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 240, dtype=np.uint8)
    for idx, item in enumerate(items):
        overlay_path = out_root / item.overlay_relpath
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        resized = cv2.resize(image, (tile_w, tile_h - 30), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, columns)
        y0 = r * tile_h
        x0 = c * tile_w
        sheet[y0 : y0 + tile_h - 30, x0 : x0 + tile_w] = resized
        caption_strip = sheet[y0 + tile_h - 30 : y0 + tile_h, x0 : x0 + tile_w]
        caption_strip[:] = BADGE_COLORS.get(item.provenance, (90, 90, 90))
        cv2.putText(
            sheet,
            f"{idx:02d}  {item.provenance}",
            (x0 + 6, y0 + tile_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    out_root.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = out_root / "report" / "contact_sheet.jpg"
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(contact_sheet_path), sheet)
    return contact_sheet_path


def build_demo_video(
    items: Sequence[DemoItem],
    out_root: Path,
    fps: float = 0.75,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path | None:
    """Stitch a slide-through MP4 of the curated overlays."""
    if not items:
        return None
    out_root.mkdir(parents=True, exist_ok=True)
    video_path = out_root / "demo_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        return None
    w, h = frame_size
    try:
        for item in items:
            overlay_path = out_root / item.overlay_relpath
            image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            canvas = np.full((h, w, 3), 30, dtype=np.uint8)
            ih, iw = image.shape[:2]
            scale = min(w / iw, (h - 60) / ih)
            new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
            cv2.putText(
                canvas,
                f"{item.stem}  |  {item.provenance}",
                (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()
    return video_path if video_path.is_file() else None


def _collect_first_n(directory: Path, pattern: str, limit: int) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob(pattern))[:limit]


def compose_demo_summary(
    items: Sequence[DemoItem],
    *,
    out_dir: Path,
    contact_sheet: Path | None,
    demo_video: Path | None,
    notes: Iterable[str],
) -> dict:
    """Build the demo_summary.json payload."""
    return {
        "demo_kind": "awe_demo_pack",
        "version": 1,
        "production_claim": False,
        "ar_ready_claim": False,
        "schema_changed": False,
        "training_performed": False,
        "provenance_labels": list(PROVENANCE_LABELS),
        "out_dir": str(out_dir),
        "contact_sheet": (
            str(contact_sheet.relative_to(out_dir)) if contact_sheet else None
        ),
        "demo_video": (str(demo_video.relative_to(out_dir)) if demo_video else None),
        "items": [item.to_dict() for item in items],
        "notes": list(notes),
    }


def build_demo_pack(
    *,
    out_dir: Path,
    pred_overlay_dir: Path = DEFAULT_PRED_OVERLAY_DIR,
    pred_json_dir: Path = DEFAULT_PRED_JSON_DIR,
    source_image_dir: Path = DEFAULT_SOURCE_IMAGE_DIR,
    annotation_preview_dir: Path = DEFAULT_ANNOTATION_PREVIEW_DIR,
    synthetic_keypoint_dir: Path = DEFAULT_SYNTHETIC_KEYPOINT_DIR,
    synthetic_pose_dir: Path = DEFAULT_SYNTHETIC_POSE_DIR,
    count: int = 15,
    annotation_count: int = 2,
    synthetic_count: int = 2,
    write_video: bool = True,
) -> dict:
    """Compose the demo pack into ``out_dir`` and return the summary dict."""
    out_dir = Path(out_dir)
    overlays_dir = out_dir / "overlays"
    json_dir = out_dir / "json"
    report_dir = out_dir / "report"
    for sub in (overlays_dir, json_dir, report_dir):
        sub.mkdir(parents=True, exist_ok=True)

    items: list[DemoItem] = []
    triples = iter_model_prediction_frames(
        pred_overlay_dir, pred_json_dir, source_image_dir, count
    )

    for pred_overlay, json_path, source_image in triples:
        stem = pred_overlay.stem.removesuffix("_pred")
        model_overlay_rel = f"overlays/{stem}__model_prediction.jpg"
        _badge_copy(pred_overlay, out_dir / model_overlay_rel, "model_prediction")
        json_rel = f"json/{stem}.json"
        _safe_copy(json_path, out_dir / json_rel)
        items.append(
            DemoItem(
                stem=stem,
                provenance="model_prediction",
                overlay_relpath=model_overlay_rel,
                source_relpath=_safe_relpath(source_image, ROOT),
                json_relpath=json_rel,
                notes=["existing baseline-inference overlay; reused, not regenerated"],
            )
        )
        source_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if source_bgr is None:
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        mock = draw_ar_mock_overlay(source_bgr, payload.get("wheels", []))
        mock_rel = f"overlays/{stem}__ar_mock.jpg"
        cv2.imwrite(str(out_dir / mock_rel), mock)
        items.append(
            DemoItem(
                stem=stem,
                provenance="ar_mock_visualization",
                overlay_relpath=mock_rel,
                source_relpath=_safe_relpath(source_image, ROOT),
                json_relpath=json_rel,
                notes=[
                    "visual explanation of the AR handoff",
                    "plane is drawn from JSON, not raycast",
                ],
            )
        )

    annotation_paths = _collect_first_n(
        annotation_preview_dir, "*_preview.jpg", annotation_count
    )
    for ann in annotation_paths:
        stem = ann.stem.removesuffix("_preview")
        rel = f"overlays/{stem}__annotation_preview.jpg"
        _badge_copy(ann, out_dir / rel, "annotation_preview")
        items.append(
            DemoItem(
                stem=stem,
                provenance="annotation_preview",
                overlay_relpath=rel,
                source_relpath=_safe_relpath(ann, ROOT),
                notes=["auto-annotation draft; not human-verified"],
            )
        )

    synthetic_paths: list[Path] = []
    synthetic_paths.extend(
        _collect_first_n(
            synthetic_keypoint_dir, "*_preview.jpg", max(1, synthetic_count - 1)
        )
    )
    synthetic_paths.extend(_collect_first_n(synthetic_pose_dir, "*_pose_labels.jpg", 1))
    for syn in synthetic_paths[:synthetic_count]:
        rel = f"overlays/{syn.stem}__synthetic_smoke.jpg"
        _badge_copy(syn, out_dir / rel, "synthetic_smoke")
        items.append(
            DemoItem(
                stem=syn.stem,
                provenance="synthetic_smoke",
                overlay_relpath=rel,
                source_relpath=_safe_relpath(syn, ROOT),
                notes=["cartoon synthetic data; plumbing only"],
            )
        )

    contact_sheet = build_contact_sheet(items, out_dir)
    demo_video = build_demo_video(items, out_dir) if write_video else None

    summary = compose_demo_summary(
        items,
        out_dir=out_dir,
        contact_sheet=contact_sheet,
        demo_video=demo_video,
        notes=[
            "Not production. Not AR-ready.",
            "Mock AR plane is drawn, not raycast.",
            "JSON sidecars are unchanged copies of outputs/demo/json/*.json.",
        ],
    )
    (out_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pred-overlay-dir", type=Path, default=DEFAULT_PRED_OVERLAY_DIR)
    p.add_argument("--pred-json-dir", type=Path, default=DEFAULT_PRED_JSON_DIR)
    p.add_argument("--source-image-dir", type=Path, default=DEFAULT_SOURCE_IMAGE_DIR)
    p.add_argument(
        "--annotation-preview-dir", type=Path, default=DEFAULT_ANNOTATION_PREVIEW_DIR
    )
    p.add_argument(
        "--synthetic-keypoint-dir", type=Path, default=DEFAULT_SYNTHETIC_KEYPOINT_DIR
    )
    p.add_argument(
        "--synthetic-pose-dir", type=Path, default=DEFAULT_SYNTHETIC_POSE_DIR
    )
    p.add_argument("--count", type=int, default=15)
    p.add_argument("--annotation-count", type=int, default=2)
    p.add_argument("--synthetic-count", type=int, default=2)
    p.add_argument("--no-video", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_demo_pack(
        out_dir=args.out_dir,
        pred_overlay_dir=args.pred_overlay_dir,
        pred_json_dir=args.pred_json_dir,
        source_image_dir=args.source_image_dir,
        annotation_preview_dir=args.annotation_preview_dir,
        synthetic_keypoint_dir=args.synthetic_keypoint_dir,
        synthetic_pose_dir=args.synthetic_pose_dir,
        count=args.count,
        annotation_count=args.annotation_count,
        synthetic_count=args.synthetic_count,
        write_video=not args.no_video,
    )
    n = len(summary["items"])
    print(f"Wrote demo pack to {args.out_dir} ({n} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

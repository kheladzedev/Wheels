"""Manual real-photo annotator for the wheel keypoint format.

Bridges the gap until the Android plugin starts sending real batches:
walks a directory of photos, opens each one in an OpenCV window, and lets
the user click

    bbox corner 1  →  bbox corner 2
    A floor/raycast point  →  B floor/raycast point
    C disc bottom

per wheel. Bbox should enclose the **full wheel including the tire**.

Semantics (2026-05-14 revision — see docs/KEYPOINT_SPEC.md):
    * A = left floor / raycast point. Screen-space pixel on the floor
      / base near the wheel's footprint, used by AR as a raycast
      source onto the floor plane. **NOT** a metal-rim edge point.
    * B = right floor / raycast point. Mirror of A on the right side
      of the wheel's footprint. **NOT** a metal-rim edge point.
    * C = lowest visible point of the metal rim / disc. NOT tire,
      NOT floor, NOT the tire/ground contact point.

Press ``a`` to add another wheel on the same image, ``n`` / ``Enter``
to save and advance, ``r`` to reset the current wheel, ``s`` to mark
the image as having no wheels, ``q`` to quit.

The annotations land incrementally in ``--annotations-dir`` (one
``<stem>.json`` per processed image) so a crash or quit doesn't lose
work. When the loop exits (any reason), images + annotations are
packaged into ``--output-root`` in the plugin's on-disk layout
(`images/`, `annotations/`, `metadata/source_info.json`) so the rest of
the pipeline (`check_keypoint_incoming.py`,
`convert_keypoint_incoming_to_yolo_pose.py`, ...) consumes the bundle
as if it came from the plugin.

The annotation JSON shape matches the plugin contract exactly:

    {
      "frame_id": "<image_stem>",
      "image":    "<filename>",
      "wheels": [
        {
          "bbox_xyxy": [x1, y1, x2, y2],
          "points": {
            "a":             [x, y],
            "b":             [x, y],
            "c_disc_bottom": [x, y]
          }
        }
      ]
    }

Usage:
    python src/manual_keypoint_annotator.py \\
        --images-dir       data/manual_real/images \\
        --annotations-dir  data/manual_real/annotations \\
        --output-root      data/incoming/manual_real \\
        --start-index      0

Pure helper functions (bbox normalisation, builders, writers) are kept
free of OpenCV imports so they can be unit-tested headlessly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_NAME = "manual_real"
SOURCE_NOTE = "manual real-photo sanity dataset"
ANNOTATION_METHOD = "manual clicks"

# Keys carried by auto-draft annotations that must be stripped before the
# cleaned annotation goes into the plugin batch — `check_keypoint_incoming.py`
# rejects unknown keys inside `points`, and the AR contract forbids per-wheel
# diagnostics. The top-level `_draft` / `_warning` / `_annotation_method` are
# dropped too so the cleaned bundle is indistinguishable from manual output.
DRAFT_WHEEL_DROP_KEYS: frozenset[str] = frozenset(
    {
        "_detector_conf",
        "_vehicle_conf",
        "_vehicle_class",
        "_mask_area_px",
        "_needs_review",
        "_review_reasons",
        "_circularity",
        "_brightness",
    }
)
DRAFT_TOPLEVEL_DROP_KEYS: frozenset[str] = frozenset(
    {"_draft", "_warning", "_annotation_method"}
)

# Drag/select hit radius in DISPLAY-space pixels. Generous so users can
# grab a 4-px keypoint marker without zooming. Tuned to be small enough
# that 3 keypoints on the same 60-px wheel are still individually
# selectable.
DRAG_HIT_RADIUS_PX = 14

# Click sequence per wheel — index into this list is the next click expected.
# Wording is the canonical UI text under the 2026-05-14 spec revision:
# A/B are floor / raycast points (screen-space raycast sources onto the
# floor plane near the wheel's footprint / base), NOT metal-rim edges.
# Do not reintroduce "rim left / rim right" / "metal rim left/right" /
# "left/right point of metal rim" wording.
CLICK_LABELS: tuple[str, str, str, str, str] = (
    "bbox corner 1",
    "bbox corner 2",
    "A floor/raycast point",
    "B floor/raycast point",
    "C disc bottom",
)

# Display-side cap. Big photos get scaled down for the window; clicks are
# rescaled back to original pixel coordinates before they enter the JSON.
DEFAULT_MAX_DISPLAY_SIDE = 1280


# ---------------------------------------------------------------------------
# Pure helpers (no OpenCV) — covered by tests/test_manual_keypoint_annotator.py
# ---------------------------------------------------------------------------


def normalize_bbox(
    corner_a: tuple[float, float] | list[float],
    corner_b: tuple[float, float] | list[float],
) -> list[float]:
    """Return ``[x1, y1, x2, y2]`` regardless of click order.

    Manual clicks happen in any direction (top-left → bottom-right is
    nominal, but a fumbled drag may put the second click anywhere); the
    plugin contract requires ``x1 < x2`` and ``y1 < y2``.
    """
    x1 = float(min(corner_a[0], corner_b[0]))
    y1 = float(min(corner_a[1], corner_b[1]))
    x2 = float(max(corner_a[0], corner_b[0]))
    y2 = float(max(corner_a[1], corner_b[1]))
    return [x1, y1, x2, y2]


def build_wheel(
    bbox_xyxy: list[float],
    point_a: list[float],
    point_b: list[float],
    point_c: list[float],
) -> dict:
    """Build one wheel entry in the plugin contract shape."""
    return {
        "bbox_xyxy": [float(v) for v in bbox_xyxy],
        "points": {
            "a": [float(point_a[0]), float(point_a[1])],
            "b": [float(point_b[0]), float(point_b[1])],
            "c_disc_bottom": [float(point_c[0]), float(point_c[1])],
        },
    }


def build_annotation(frame_id: str, image_name: str, wheels: list[dict]) -> dict:
    """Build the per-image annotation JSON (plugin contract)."""
    return {
        "frame_id": str(frame_id),
        "image": str(image_name),
        "wheels": list(wheels),
    }


def build_source_info() -> dict:
    """Metadata documenting the batch's provenance for the converter."""
    return {
        "source_name": SOURCE_NAME,
        "note": SOURCE_NOTE,
        "annotation_method": ANNOTATION_METHOD,
    }


def strip_draft_wheel(wheel: dict) -> dict:
    """Return a clean copy of ``wheel`` with auto-draft diagnostic keys removed.

    Preserves ``bbox_xyxy`` + ``points`` exactly (no rounding) so a "press
    y to accept" pass is lossless. Unknown extra keys outside the drop
    list are kept verbatim — future-compat — but the plugin validator
    will flag them, so callers should not rely on this.
    """
    cleaned: dict = {}
    for key, value in wheel.items():
        if key in DRAFT_WHEEL_DROP_KEYS:
            continue
        cleaned[key] = value
    points = cleaned.get("points")
    if isinstance(points, dict):
        cleaned["points"] = {
            name: [float(xy[0]), float(xy[1])]
            for name, xy in points.items()
            if name in {"a", "b", "c_disc_bottom"}
            and isinstance(xy, (list, tuple))
            and len(xy) == 2
        }
    bbox = cleaned.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        cleaned["bbox_xyxy"] = [float(v) for v in bbox]
    return cleaned


def load_draft_wheels(path: Path) -> list[dict]:
    """Read a draft JSON, return a clean ``wheels`` list ready for editing.

    Returns ``[]`` if the file is missing, not JSON, missing/empty ``wheels``,
    or any wheel lacks the required shape. The function is forgiving on
    purpose — auto-drafts sometimes ship partial outputs, and the user
    should still be able to enter the image and re-click from scratch.
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    raw_wheels = payload.get("wheels")
    if not isinstance(raw_wheels, list):
        return []

    cleaned: list[dict] = []
    for w in raw_wheels:
        if not isinstance(w, dict):
            continue
        bbox = w.get("bbox_xyxy")
        points = w.get("points")
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and isinstance(points, dict)
            and {"a", "b", "c_disc_bottom"}.issubset(points.keys())
        ):
            continue
        cleaned.append(strip_draft_wheel(w))
    return cleaned


def find_hit_keypoint(
    wheels: list[dict],
    display_xy: tuple[float, float] | list[float],
    scale: float,
    hit_radius_px: float = DRAG_HIT_RADIUS_PX,
) -> tuple[int, str] | None:
    """Return ``(wheel_idx, point_name)`` for the closest keypoint within
    ``hit_radius_px`` in display space, or ``None``.

    Ties broken by iteration order so behaviour is deterministic on
    overlapping keypoints (rare, but happens on very small wheels).
    """
    if scale <= 0:
        return None
    dx_ref = float(display_xy[0])
    dy_ref = float(display_xy[1])
    r2 = float(hit_radius_px) * float(hit_radius_px)
    best: tuple[int, str] | None = None
    best_d2 = r2
    for w_idx, wheel in enumerate(wheels):
        points = wheel.get("points")
        if not isinstance(points, dict):
            continue
        for name in ("a", "b", "c_disc_bottom"):
            xy = points.get(name)
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                continue
            dx = float(xy[0]) * scale - dx_ref
            dy = float(xy[1]) * scale - dy_ref
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best = (w_idx, name)
                best_d2 = d2
    return best


def find_hit_bbox(
    wheels: list[dict],
    display_xy: tuple[float, float] | list[float],
    scale: float,
) -> int | None:
    """Return the index of the wheel whose bbox contains ``display_xy``,
    or ``None``. If multiple bboxes overlap the point, the smallest-area
    bbox wins (so a tiny wheel sitting on top of a big one stays grabbable).
    """
    if scale <= 0:
        return None
    dx = float(display_xy[0])
    dy = float(display_xy[1])
    best_idx: int | None = None
    best_area = float("inf")
    for w_idx, wheel in enumerate(wheels):
        bbox = wheel.get("bbox_xyxy")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1 = float(bbox[0]) * scale
        y1 = float(bbox[1]) * scale
        x2 = float(bbox[2]) * scale
        y2 = float(bbox[3]) * scale
        if x1 <= dx <= x2 and y1 <= dy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area < best_area:
                best_area = area
                best_idx = w_idx
    return best_idx


def apply_keypoint_drag(
    wheels: list[dict],
    wheel_idx: int,
    point_name: str,
    new_xy_image: tuple[float, float] | list[float],
) -> list[dict]:
    """Return a copy of ``wheels`` with one keypoint moved.

    New coordinates are stored in **image space**, not display space.
    Callers must convert before invoking.
    """
    if not (0 <= wheel_idx < len(wheels)):
        raise IndexError(f"wheel_idx {wheel_idx} out of range (len={len(wheels)})")
    if point_name not in {"a", "b", "c_disc_bottom"}:
        raise ValueError(
            f"point_name must be one of a/b/c_disc_bottom, got {point_name!r}"
        )

    out = [dict(w) for w in wheels]
    target = dict(out[wheel_idx])
    new_points = dict(target.get("points") or {})
    new_points[point_name] = [float(new_xy_image[0]), float(new_xy_image[1])]
    target["points"] = new_points
    out[wheel_idx] = target
    return out


def write_annotation(path: Path, annotation: dict) -> None:
    """Atomic-ish write so a Ctrl-C mid-write doesn't leave a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_source_info(meta_dir: Path) -> Path:
    """Write ``meta_dir/source_info.json`` and return its path."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "source_info.json"
    out.write_text(json.dumps(build_source_info(), indent=2), encoding="utf-8")
    return out


def list_images(images_dir: Path, start_index: int = 0) -> list[Path]:
    """Sorted list of image files in ``images_dir`` from ``start_index``."""
    if not images_dir.is_dir():
        return []
    all_images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if start_index < 0:
        start_index = 0
    return all_images[start_index:]


def scale_for_display(img_h: int, img_w: int, max_side: int) -> float:
    """Scale factor s ∈ (0, 1] such that scaled ``max(w, h) ≤ max_side``.

    Used so big photos (12 MP phone shots) fit on screen, while clicks are
    still recorded in original-pixel coordinates by inverse-scaling.
    """
    longer = max(int(img_h), int(img_w))
    if longer <= 0 or longer <= max_side:
        return 1.0
    return float(max_side) / float(longer)


def display_to_image_coord(
    display_xy: tuple[float, float] | list[float], scale: float
) -> list[float]:
    """Inverse of the display-scale transform. ``scale`` is the same factor
    `scale_for_display` returned (so the original-pixel coord is
    ``display / scale``).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    return [float(display_xy[0]) / scale, float(display_xy[1]) / scale]


def package_output(
    images_dir: Path,
    annotations_dir: Path,
    output_root: Path,
) -> dict:
    """Mirror ``images_dir`` + ``annotations_dir`` into the plugin's on-disk
    layout under ``output_root``.

    Only images that have a matching ``<stem>.json`` in ``annotations_dir``
    are copied — unfinished work stays out of the convert step. Returns a
    summary dict with counts so the CLI can print it.
    """
    out_images = output_root / "images"
    out_annos = output_root / "annotations"
    out_meta = output_root / "metadata"
    for d in (out_images, out_annos, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_annotations = 0
    if images_dir.is_dir():
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            anno_path = annotations_dir / f"{img_path.stem}.json"
            if not anno_path.exists():
                continue
            shutil.copy2(img_path, out_images / img_path.name)
            shutil.copy2(anno_path, out_annos / anno_path.name)
            n_images += 1
            n_annotations += 1
    source_info_path = write_source_info(out_meta)
    return {
        "images_copied": n_images,
        "annotations_copied": n_annotations,
        "source_info": str(source_info_path),
        "output_root": str(output_root),
    }


# ---------------------------------------------------------------------------
# GUI loop (OpenCV) — not directly unit-tested, but kept thin
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual real-photo wheel keypoint annotator"
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/manual_real/images"),
        help="Directory of real photos to annotate.",
    )
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/manual_real/annotations"),
        help="Where per-image annotation JSONs are written incrementally.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/incoming/manual_real"),
        help="Plugin-layout bundle written on exit (images/ + annotations/ + "
        "metadata/source_info.json).",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N images in --images-dir (alphabetical).",
    )
    p.add_argument(
        "--max-display-side",
        type=int,
        default=DEFAULT_MAX_DISPLAY_SIDE,
        help="Maximum side length (px) of the displayed image. Clicks are "
        "always recorded in original-pixel coordinates.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="If set, re-annotate images that already have a JSON in "
        "--annotations-dir (default: skip them).",
    )
    p.add_argument(
        "--prefill-from",
        type=Path,
        default=None,
        help="Optional directory of draft annotation JSONs (e.g. output of "
        "auto_annotate_wheels.py). When present, each image opens with "
        "its draft wheels already drawn; press y/Enter to accept, drag "
        "any keypoint to fix it, click inside a bbox + d to drop a "
        "wheel, e to clear all and re-click from scratch.",
    )
    return p.parse_args(argv)


def _render(
    canvas,
    wheels: list[dict],
    current_clicks: list,
    scale: float,
    status: str,
    selected_idx: int | None = None,
) -> None:
    """Draw committed wheels (green; selected = red) + in-progress clicks
    (yellow) + status."""
    import cv2  # local import keeps headless tests fast

    h, w = canvas.shape[:2]

    # Committed wheels — full green, except the selected one in red.
    for w_idx, wheel in enumerate(wheels):
        x1, y1, x2, y2 = (int(round(v * scale)) for v in wheel["bbox_xyxy"])
        bbox_color = (0, 0, 255) if w_idx == selected_idx else (0, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.putText(
            canvas,
            f"#{w_idx}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bbox_color,
            1,
            cv2.LINE_AA,
        )
        for name, color in (
            ("a", (0, 255, 0)),
            ("b", (0, 255, 255)),
            ("c_disc_bottom", (0, 0, 255)),
        ):
            xy = wheel["points"][name]
            cv2.circle(
                canvas,
                (int(round(xy[0] * scale)), int(round(xy[1] * scale))),
                4,
                color,
                -1,
            )

    # Current-wheel-in-progress clicks — yellow dots.
    for i, click in enumerate(current_clicks):
        cv2.circle(canvas, (int(click[0]), int(click[1])), 5, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            CLICK_LABELS[i].split(" ", 1)[0],
            (int(click[0]) + 6, int(click[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # If both bbox corners are placed, show the bbox preview.
    if len(current_clicks) >= 2:
        x1 = int(min(current_clicks[0][0], current_clicks[1][0]))
        y1 = int(min(current_clicks[0][1], current_clicks[1][1]))
        x2 = int(max(current_clicks[0][0], current_clicks[1][0]))
        y2 = int(max(current_clicks[0][1], current_clicks[1][1]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Status line at the top.
    cv2.rectangle(canvas, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _annotate_image(
    image_path: Path,
    max_display_side: int,
    prefill_wheels: list[dict] | None = None,
) -> list[dict] | str:
    """Run the click UI for one image. Returns either:

    - list[dict]: the wheels list (possibly empty) to save.
    - "skip":     user pressed `s` — save empty wheels.
    - "quit":     user pressed `q` — caller should finalise + exit.

    If ``prefill_wheels`` is non-empty, those wheels are drawn from the
    start. The user can drag any keypoint to fix it, click inside a bbox
    + press ``d`` to drop a wheel, or press ``e`` to clear all wheels
    and re-click from scratch. Adding extra wheels on top of the
    prefilled set still works via the existing 5-click sequence.
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"WARN: cannot decode {image_path}, skipping")
        return "skip"

    h, w = img.shape[:2]
    scale = scale_for_display(h, w, max_display_side)
    disp_size = (int(round(w * scale)), int(round(h * scale)))

    window = "manual_keypoint_annotator"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    wheels: list[dict] = [dict(w) for w in (prefill_wheels or [])]
    current_clicks: list[tuple[float, float]] = []  # display-space
    selected_wheel_idx: list[int | None] = [None]

    # Drag state. Boxed in a list so the nested mouse callback can mutate
    # it without nonlocal gymnastics. Layout: [active, wheel_idx, point_name].
    drag_state: list = [False, None, None]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Drag takes priority over fresh clicks if the cursor is on a
            # keypoint. Selection takes priority over fresh clicks if the
            # cursor is inside a bbox but not on a keypoint. Otherwise
            # fall through to the canonical 5-click sequence.
            hit_kp = find_hit_keypoint(wheels, (x, y), scale)
            if hit_kp is not None:
                drag_state[0] = True
                drag_state[1] = hit_kp[0]
                drag_state[2] = hit_kp[1]
                return
            hit_bbox = find_hit_bbox(wheels, (x, y), scale)
            if hit_bbox is not None and not current_clicks:
                selected_wheel_idx[0] = hit_bbox
                return
            if len(current_clicks) < len(CLICK_LABELS):
                current_clicks.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drag_state[0]:
            new_img_xy = display_to_image_coord((x, y), scale)
            wheels[drag_state[1]]["points"][drag_state[2]] = [
                float(new_img_xy[0]),
                float(new_img_xy[1]),
            ]
        elif event == cv2.EVENT_LBUTTONUP and drag_state[0]:
            drag_state[0] = False
            drag_state[1] = None
            drag_state[2] = None

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = cv2.resize(img, disp_size, interpolation=cv2.INTER_AREA)
        next_label = (
            CLICK_LABELS[len(current_clicks)]
            if len(current_clicks) < len(CLICK_LABELS)
            else "complete — press n/Enter or a"
        )
        sel = selected_wheel_idx[0]
        sel_info = f" selected:#{sel}" if sel is not None else ""
        status = (
            f"{image_path.name} | wheels: {len(wheels)}{sel_info} | "
            f"click {len(current_clicks)}/{len(CLICK_LABELS)}: {next_label} | "
            "n/Enter=save  d=drop-selected  e=clear-all  a=add  r=reset  s=skip  q=quit"
        )
        _render(display, wheels, current_clicks, scale, status, selected_idx=sel)
        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        if key in (ord("q"),):
            cv2.destroyWindow(window)
            return "quit"
        if key == ord("r"):
            current_clicks.clear()
            continue
        if key == ord("s"):
            cv2.destroyWindow(window)
            return "skip"
        if key == ord("d"):
            sel = selected_wheel_idx[0]
            if sel is not None and 0 <= sel < len(wheels):
                dropped = wheels.pop(sel)
                selected_wheel_idx[0] = None
                print(f"INFO: dropped wheel #{sel} (bbox {dropped.get('bbox_xyxy')})")
            else:
                print(
                    "INFO: no wheel selected — click inside a bbox first, then press d."
                )
            continue
        if key == ord("e"):
            if wheels:
                print(f"INFO: cleared {len(wheels)} pre-filled wheel(s)")
            wheels.clear()
            selected_wheel_idx[0] = None
            current_clicks.clear()
            continue
        if key in (ord("n"), ord("y"), 13, 10):  # n / y / Enter (CR/LF)
            if len(current_clicks) == len(CLICK_LABELS):
                wheels.append(_finalise_wheel(current_clicks, scale))
                current_clicks.clear()
            elif len(current_clicks) > 0:
                # Partial wheel — refuse, keep state.
                print(
                    f"INFO: {len(current_clicks)}/{len(CLICK_LABELS)} clicks "
                    "placed — finish the wheel or press r to reset."
                )
                continue
            cv2.destroyWindow(window)
            return wheels
        if key == ord("a"):
            if len(current_clicks) == len(CLICK_LABELS):
                wheels.append(_finalise_wheel(current_clicks, scale))
                current_clicks.clear()
                print(f"INFO: saved wheel #{len(wheels) - 1}, add another")
            else:
                print(
                    f"INFO: finish the current wheel ({len(current_clicks)}/"
                    f"{len(CLICK_LABELS)} clicks) before adding another."
                )
            continue


def _finalise_wheel(display_clicks: list[tuple[float, float]], scale: float) -> dict:
    """Convert the 5 display-space clicks into one plugin-shaped wheel."""
    img_clicks = [display_to_image_coord(c, scale) for c in display_clicks]
    bbox = normalize_bbox(img_clicks[0], img_clicks[1])
    return build_wheel(bbox, img_clicks[2], img_clicks[3], img_clicks[4])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.images_dir.is_dir():
        print(f"ERROR: images dir not found: {args.images_dir}")
        return 2

    args.annotations_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images_dir, start_index=args.start_index)
    if not images:
        print(f"ERROR: no images in {args.images_dir} from index {args.start_index}")
        return 1

    prefill_dir: Path | None = args.prefill_from
    if prefill_dir is not None and not prefill_dir.is_dir():
        print(f"ERROR: --prefill-from is not a directory: {prefill_dir}")
        return 2

    print(f"Annotating {len(images)} image(s) from {args.images_dir}")
    print(f"Annotations land in:  {args.annotations_dir}")
    print(f"Output bundle root:   {args.output_root}")
    if prefill_dir is not None:
        print(f"Prefill drafts from:  {prefill_dir}")
        print(
            "Keys: y/n/Enter = save+next, d = drop selected wheel, "
            "e = clear all, a = add wheel, r = reset, s = skip, q = quit"
        )
    else:
        print("Keys: n/Enter = save+next, a = add wheel, r = reset, s = skip, q = quit")

    processed = 0
    quit_requested = False
    for img_path in images:
        anno_path = args.annotations_dir / f"{img_path.stem}.json"
        if anno_path.exists() and not args.rerun:
            print(f"SKIP (already annotated): {img_path.name}")
            continue

        prefill_wheels: list[dict] = []
        if prefill_dir is not None:
            draft_path = prefill_dir / f"{img_path.stem}.json"
            prefill_wheels = load_draft_wheels(draft_path)
            if prefill_wheels:
                print(
                    f"PREFILL: {draft_path.name} -> "
                    f"{len(prefill_wheels)} draft wheel(s)"
                )

        result = _annotate_image(
            img_path, args.max_display_side, prefill_wheels=prefill_wheels
        )
        if result == "quit":
            quit_requested = True
            break
        if result == "skip":
            wheels: list[dict] = []
        else:
            wheels = result

        annotation = build_annotation(img_path.stem, img_path.name, wheels)
        write_annotation(anno_path, annotation)
        processed += 1
        print(f"SAVED: {anno_path.name} ({len(wheels)} wheel(s))")

    summary = package_output(args.images_dir, args.annotations_dir, args.output_root)
    print()
    print("Session summary:")
    print(f"  Processed this run:  {processed}")
    print(f"  Quit requested:      {quit_requested}")
    print("Output bundle:")
    print(f"  images_copied:       {summary['images_copied']}")
    print(f"  annotations_copied:  {summary['annotations_copied']}")
    print(f"  source_info:         {summary['source_info']}")
    print(f"  output_root:         {summary['output_root']}")
    print()
    print("Next: validate the bundle:")
    print(f"  python src/check_keypoint_incoming.py --source-root {args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

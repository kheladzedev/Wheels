"""Inspect a raw Unreal/plugin export without converting it.

Read-only inspector. Builds a per-object status table and a handful of
preview JPEGs so the team can decide how (and whether) to ingest the
batch. Does **not** write to ``data/incoming/...``, does **not** train,
does **not** run inference.

Expected layout::

    <source-root>/
        Images/<frame_id>.jpg
        Ground/<frame_id>.txt        (optional metadata)
        keyPoint/<frame_id>/<object_id>.txt

Per-object status:
    ``VALID_ALL_POINTS_IN_IMAGE``
    ``EMPTY_ALL_ZERO``
    ``PARTIAL_ZERO``
    ``OUT_OF_BOUNDS``
    ``MISSING_POINTS``
    ``PARSE_ERROR``

Outputs::

    <out-dir>/report.json
    <out-dir>/report.md
    <out-dir>/previews/<frame_id>.jpg

Usage::

    python scripts/inspect_unreal_export.py \\
        --source-root ~/Downloads/0001 \\
        --out-dir outputs/unreal_export_inspection \\
        --max-preview 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
POINT_NAMES = ("Right", "Left", "Center")
OPTIONAL_POINT_NAMES = ("LeftTop", "RightTop")
ALL_POINT_NAMES = POINT_NAMES + OPTIONAL_POINT_NAMES
POINT_NAME_ALIASES = {
    "SphereRight": "Right",
    "SphereLeft": "Left",
    "SphereRightTop": "RightTop",
    "SphereLeftTop": "LeftTop",
}
RAW_POINT_NAMES = ALL_POINT_NAMES + tuple(POINT_NAME_ALIASES)
_POINT_NAME_RE = "|".join(sorted(RAW_POINT_NAMES, key=len, reverse=True))
ZERO_EPS = 1e-6

# Two formats observed in the wild — be tolerant.
#   Unreal:  {name:"Right",XY:1.0,2.0},
#   Simple:  Right: 1.0,2.0
UNREAL_RE = re.compile(
    rf'name\s*:\s*"({_POINT_NAME_RE})"\s*,\s*'
    r"XY\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)
SIMPLE_RE = re.compile(
    rf"^\s*({_POINT_NAME_RE})\s*:"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)

# DeltaZ{170.000019},Roll:-0.0,Pitch:61.769783,FOV:54.656362
GROUND_RE = re.compile(
    r"DeltaZ\s*\{?\s*(-?\d+(?:\.\d+)?)\s*\}?\s*"
    r",\s*Roll\s*:\s*(-?\d+(?:\.\d+)?)\s*"
    r",\s*Pitch\s*:\s*(-?\d+(?:\.\d+)?)\s*"
    r",\s*FOV\s*:\s*(-?\d+(?:\.\d+)?)"
)

STATUS_VALID = "VALID_ALL_POINTS_IN_IMAGE"
STATUS_EMPTY = "EMPTY_ALL_ZERO"
STATUS_PARTIAL_ZERO = "PARTIAL_ZERO"
STATUS_OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
STATUS_MISSING = "MISSING_POINTS"
STATUS_PARSE_ERROR = "PARSE_ERROR"

ALL_STATUSES = (
    STATUS_VALID,
    STATUS_EMPTY,
    STATUS_PARTIAL_ZERO,
    STATUS_OUT_OF_BOUNDS,
    STATUS_MISSING,
    STATUS_PARSE_ERROR,
)

# BGR — explicit distinct colors so Right/Left/Center are unambiguous
# even when two points sit near each other.
COLOR_RIGHT = (255, 0, 0)
COLOR_LEFT = (0, 255, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_LEFT_TOP = (0, 220, 255)
COLOR_RIGHT_TOP = (255, 0, 255)
COLOR_MAP = {
    "Right": COLOR_RIGHT,
    "Left": COLOR_LEFT,
    "Center": COLOR_CENTER,
    "LeftTop": COLOR_LEFT_TOP,
    "RightTop": COLOR_RIGHT_TOP,
}


@dataclass
class GroundMeta:
    delta_z: float
    roll: float
    pitch: float
    fov: float


@dataclass
class KeyPointObject:
    frame_id: str
    object_id: str
    file: Path
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    status: str = STATUS_PARSE_ERROR
    reason: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect a raw Unreal/plugin export (no conversion, no training)."
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/unreal_export_inspection"),
    )
    p.add_argument("--max-preview", type=int, default=30)
    p.add_argument(
        "--max-status-preview",
        type=int,
        default=10,
        help="How many per-status frames to render under previews/by_status/<status>/.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------- parsing -----------------------------------------------------------


def parse_keypoint_text(text: str) -> dict[str, tuple[float, float]]:
    """Parse a keyPoint .txt body.

    Tolerant of the two formats seen in real exports. Returns a dict
    ``{Right|Left|Center|LeftTop|RightTop: (x, y)}``. Unreal Blueprint
    point actor names from Igor's docs (``SphereLeft``/``SphereRight`` and
    top variants) are normalized to those canonical names. Names not found
    are absent. Only Right/Left/Center are required for the legacy status
    classification; LeftTop/RightTop are optional bbox helpers observed in
    newer trial exports.
    """
    out: dict[str, tuple[float, float]] = {}
    for m in UNREAL_RE.finditer(text):
        name = POINT_NAME_ALIASES.get(m.group(1), m.group(1))
        out.setdefault(name, (float(m.group(2)), float(m.group(3))))
    if not out:
        for m in SIMPLE_RE.finditer(text):
            name = POINT_NAME_ALIASES.get(m.group(1), m.group(1))
            out.setdefault(name, (float(m.group(2)), float(m.group(3))))
    return out


def parse_ground_text(text: str) -> Optional[GroundMeta]:
    m = GROUND_RE.search(text)
    if not m:
        return None
    try:
        return GroundMeta(
            delta_z=float(m.group(1)),
            roll=float(m.group(2)),
            pitch=float(m.group(3)),
            fov=float(m.group(4)),
        )
    except ValueError:
        return None


# ---------- classification ----------------------------------------------------


def classify(
    points: dict[str, tuple[float, float]],
    image_w: int,
    image_h: int,
) -> tuple[str, str]:
    """Return ``(status, reason)``.

    Precedence: ``MISSING_POINTS`` → ``EMPTY_ALL_ZERO`` →
    ``PARTIAL_ZERO`` → ``OUT_OF_BOUNDS`` → ``VALID_ALL_POINTS_IN_IMAGE``.
    """
    missing = [n for n in POINT_NAMES if n not in points]
    if missing:
        return STATUS_MISSING, f"missing: {','.join(missing)}"

    zeros = [
        n
        for n in POINT_NAMES
        if abs(points[n][0]) <= ZERO_EPS and abs(points[n][1]) <= ZERO_EPS
    ]
    if len(zeros) == len(POINT_NAMES):
        return STATUS_EMPTY, "all three points are (0, 0)"
    if zeros:
        return STATUS_PARTIAL_ZERO, f"zero: {','.join(zeros)}"

    oob = [
        n
        for n in POINT_NAMES
        if not (0 <= points[n][0] <= image_w - 1 and 0 <= points[n][1] <= image_h - 1)
    ]
    if oob:
        details = ",".join(f"{n}=({points[n][0]:.1f},{points[n][1]:.1f})" for n in oob)
        return STATUS_OUT_OF_BOUNDS, f"outside image: {details}"

    return STATUS_VALID, ""


# ---------- JPEG-header size reader (no full decode) --------------------------


def read_jpeg_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a JPEG SOF marker, or None if unavailable.

    Avoids decoding pixels. Falls back to ``cv2.imread`` in the caller if needed.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9):  # SOI / EOI
            i += 2
            continue
        if marker == 0xDA:  # SOS — scan; SOFs always come before this
            return None
        # SOF0..SOF15 except DHT(0xC4), JPG(0xC8), DAC(0xCC)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 >= n:
                return None
            h = (data[i + 5] << 8) + data[i + 6]
            w = (data[i + 7] << 8) + data[i + 8]
            return w, h
        if i + 4 > n:
            return None
        seg_len = (data[i + 2] << 8) + data[i + 3]
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def image_size(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        size = read_jpeg_size(path)
        if size is not None:
            return size
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


# ---------- discovery ---------------------------------------------------------


def discover_frames(source_root: Path) -> dict[str, dict[str, Path]]:
    """Group export files by frame_id.

    Each value contains zero or more of ``image`` / ``ground`` / ``kp_dir``.
    """
    frames: dict[str, dict[str, Path]] = {}

    images_dir = source_root / "Images"
    if images_dir.is_dir():
        for img in images_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTS and img.is_file():
                frames.setdefault(img.stem, {})["image"] = img

    ground_dir = source_root / "Ground"
    if ground_dir.is_dir():
        for g in ground_dir.iterdir():
            if g.suffix.lower() == ".txt" and g.is_file():
                frames.setdefault(g.stem, {})["ground"] = g

    kp_dir = source_root / "keyPoint"
    if kp_dir.is_dir():
        for d in kp_dir.iterdir():
            if d.is_dir():
                frames.setdefault(d.name, {})["kp_dir"] = d

    return frames


# ---------- preview rendering -------------------------------------------------


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
) -> None:
    """Outlined text: black halo + colored fill, so it survives any background."""
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_preview(
    img_path: Path,
    objects_for_frame: list[KeyPointObject],
    out_path: Path,
) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]

    _put_text(
        img,
        "RAW EXPORT: raw Right/Left/Center labels",
        (12, 36),
        (255, 255, 255),
        scale=0.9,
    )

    for obj in objects_for_frame:
        usable = any(
            not (abs(x) <= ZERO_EPS and abs(y) <= ZERO_EPS)
            for (x, y) in obj.points.values()
        )
        if not usable:
            continue

        anchor: tuple[int, int] | None = None
        for name in ALL_POINT_NAMES:
            if name not in obj.points:
                continue
            x, y = obj.points[name]
            if abs(x) <= ZERO_EPS and abs(y) <= ZERO_EPS:
                continue
            cx = max(0, min(w - 1, int(round(x))))
            cy = max(0, min(h - 1, int(round(y))))
            in_image = 0 <= x <= w - 1 and 0 <= y <= h - 1
            color = COLOR_MAP[name]
            if not in_image:
                cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 28, 3)
            else:
                cv2.circle(img, (cx, cy), 8, color, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 10, (0, 0, 0), 1, cv2.LINE_AA)
            label = name[:1] + ("*" if not in_image else "")
            _put_text(img, label, (cx + 10, cy - 10), color, scale=0.6)
            if anchor is None:
                anchor = (cx, cy)

        if anchor is not None:
            ax, ay = anchor
            tag = f"id={obj.object_id} [{obj.status}]"
            _put_text(img, tag, (ax + 14, ay + 24), (255, 255, 255), scale=0.55)

    cv2.imwrite(str(out_path), img)
    return True


# ---------- main inspection pass ----------------------------------------------


def inspect(args: argparse.Namespace) -> dict:
    src = args.source_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    previews_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(src)

    n_images = sum("image" in v for v in frames.values())
    n_ground = sum("ground" in v for v in frames.values())
    n_kp_dirs = sum("kp_dir" in v for v in frames.values())

    image_resolutions: list[tuple[int, int]] = []
    frame_size: dict[str, tuple[int, int]] = {}
    ground_parsed = 0

    objects: list[KeyPointObject] = []

    for frame_id, paths in sorted(frames.items()):
        img_path = paths.get("image")
        if img_path is not None:
            size = image_size(img_path)
            if size is not None:
                frame_size[frame_id] = size
                image_resolutions.append(size)

        ground_path = paths.get("ground")
        if ground_path is not None:
            try:
                txt = ground_path.read_text(errors="replace")
                if parse_ground_text(txt) is not None:
                    ground_parsed += 1
            except OSError:
                pass

        kp_dir = paths.get("kp_dir")
        if kp_dir is None:
            continue

        size = frame_size.get(frame_id)
        for kp_file in sorted(kp_dir.iterdir(), key=lambda p: p.name):
            if kp_file.suffix.lower() != ".txt" or not kp_file.is_file():
                continue
            obj = KeyPointObject(
                frame_id=frame_id, object_id=kp_file.stem, file=kp_file
            )
            try:
                text = kp_file.read_text(errors="replace")
                obj.points = parse_keypoint_text(text)
                if size is None:
                    obj.status = STATUS_PARSE_ERROR
                    obj.reason = "no image found for this frame_id"
                else:
                    w_, h_ = size
                    obj.status, obj.reason = classify(obj.points, w_, h_)
            except OSError as e:
                obj.status = STATUS_PARSE_ERROR
                obj.reason = f"OS error: {e}"
            objects.append(obj)

    counts = {s: 0 for s in ALL_STATUSES}
    for o in objects:
        counts[o.status] = counts.get(o.status, 0) + 1

    examples: dict[str, list[dict]] = {}
    for status in ALL_STATUSES:
        picks = [o for o in objects if o.status == status][:5]
        examples[status] = [
            {
                "frame_id": o.frame_id,
                "object_id": o.object_id,
                "file": str(o.file.relative_to(src))
                if o.file.is_relative_to(src)
                else str(o.file),
                "points": {k: [v[0], v[1]] for k, v in o.points.items()},
                "reason": o.reason,
            }
            for o in picks
        ]

    res_counter: dict[str, int] = {}
    for w_, h_ in image_resolutions:
        res_counter[f"{w_}x{h_}"] = res_counter.get(f"{w_}x{h_}", 0) + 1

    rng = random.Random(args.seed)
    by_frame: dict[str, list[KeyPointObject]] = {}
    for o in objects:
        by_frame.setdefault(o.frame_id, []).append(o)
    candidates = [fid for fid in by_frame if frames.get(fid, {}).get("image")]
    rng.shuffle(candidates)

    # Front-load frames that contain at least one VALID or OUT_OF_BOUNDS object,
    # so the preview set isn't all-zero noise.
    def _has_signal(fid: str) -> bool:
        return any(
            o.status in (STATUS_VALID, STATUS_OUT_OF_BOUNDS, STATUS_PARTIAL_ZERO)
            for o in by_frame[fid]
        )

    candidates.sort(key=lambda fid: 0 if _has_signal(fid) else 1)

    drawn: list[str] = []
    for fid in candidates:
        if len(drawn) >= args.max_preview:
            break
        img_path = frames[fid].get("image")
        out_path = previews_dir / f"{fid}.jpg"
        if img_path is not None and draw_preview(img_path, by_frame[fid], out_path):
            drawn.append(str(out_path.relative_to(out_dir)))

    status_previews: dict[str, list[str]] = {}
    by_status_root = previews_dir / "by_status"
    for status in ALL_STATUSES:
        if args.max_status_preview <= 0:
            status_previews[status] = []
            continue
        status_fids = [
            fid for fid in candidates if any(o.status == status for o in by_frame[fid])
        ]
        status_drawn: list[str] = []
        status_dir = by_status_root / status
        for fid in status_fids[: args.max_status_preview]:
            img_path = frames[fid].get("image")
            if img_path is None:
                continue
            status_dir.mkdir(parents=True, exist_ok=True)
            out_path = status_dir / f"{fid}.jpg"
            if draw_preview(img_path, by_frame[fid], out_path):
                status_drawn.append(str(out_path.relative_to(out_dir)))
        status_previews[status] = status_drawn

    report = {
        "source_root": str(src),
        "n_images": n_images,
        "n_ground_files": n_ground,
        "n_ground_parsed": ground_parsed,
        "n_keypoint_frame_dirs": n_kp_dirs,
        "n_keypoint_object_files": len(objects),
        "counts_by_status": counts,
        "image_resolutions": res_counter,
        "examples_by_status": examples,
        "previews": drawn,
        "status_previews": status_previews,
        "raw_point_aliases": POINT_NAME_ALIASES,
        "contract_notes": [
            "Right/Left raw naming differs between observed batches. The "
            "importer resolves A/B mapping from screen-space x-order by "
            "default; Center maps to points.c_disc_bottom.",
            "SphereLeft/SphereRight/SphereLeftTop/SphereRightTop are accepted "
            "as Igor Blueprint aliases and normalized before classification.",
            "LeftTop/RightTop are optional bbox helper points observed in "
            "trial exports.",
            "(0,0), missing required points, and out-of-image required "
            "points are treated as invisible/invalid and dropped by import.",
            "Ground metadata is preserved in the import report only; ML "
            "training and inference stay on image pixels + 2D points.",
        ],
        "training_acceptance": (
            "NOT_APPROVED_FOR_TRAINING — run import/validation/conversion "
            "and manually inspect previews before accepting a full batch."
        ),
    }

    write_json(out_dir / "report.json", report)
    write_md(out_dir / "report.md", report)
    return report


def write_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def write_md(path: Path, report: dict) -> None:
    lines: list[str] = []
    lines += [
        "# Raw Unreal export inspection",
        "",
        f"Source root: `{report['source_root']}`",
        "",
        "## Top-level counts",
        "",
        f"- Images: {report['n_images']}",
        f"- Ground files: {report['n_ground_files']} "
        f"(parsed OK: {report['n_ground_parsed']})",
        f"- keyPoint frame folders: {report['n_keypoint_frame_dirs']}",
        f"- keyPoint object files: {report['n_keypoint_object_files']}",
        "",
        "## Counts by status",
        "",
    ]
    for status, c in sorted(
        report["counts_by_status"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        lines.append(f"- `{status}`: {c}")
    lines.append("")

    lines += ["## Image resolution stats", ""]
    if not report["image_resolutions"]:
        lines.append("- (no images found)")
    for res, c in sorted(report["image_resolutions"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{res}`: {c}")
    lines.append("")

    lines += ["## Examples per status", ""]
    for status in ALL_STATUSES:
        exs = report["examples_by_status"].get(status, [])
        if not exs:
            continue
        lines += [f"### {status}", ""]
        for ex in exs:
            pt_str = ", ".join(
                f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in ex["points"].items()
            )
            reason = f" — {ex['reason']}" if ex.get("reason") else ""
            lines.append(
                f"- `{ex['file']}` (frame {ex['frame_id']}, "
                f"obj {ex['object_id']}): {pt_str}{reason}"
            )
        lines.append("")

    lines += [
        "## Previews",
        "",
        f"{len(report['previews'])} preview images in `previews/`. Overlay "
        "title is **RAW EXPORT: raw Right/Left/Center labels**. "
        "Point colors:",
        "",
        "- Right — blue",
        "- Left — green",
        "- Center — red",
        "- LeftTop — yellow/cyan",
        "- RightTop — magenta",
        "",
        "Out-of-image points are clamped to the image edge, drawn with a "
        "tilted cross, and labelled with a trailing `*`.",
        "",
    ]
    aliases = report.get("raw_point_aliases") or {}
    if aliases:
        alias_text = ", ".join(f"{k}->{v}" for k, v in sorted(aliases.items()))
        lines += ["Raw point aliases accepted: `" + alias_text + "`.", ""]
    lines += ["## Status preview galleries", ""]
    for status in ALL_STATUSES:
        paths = report.get("status_previews", {}).get(status, [])
        if not paths:
            continue
        lines.append(
            f"- `{status}`: {len(paths)} frame(s) under "
            f"`previews/by_status/{status}/`"
        )
    lines.append("")
    lines += ["## Contract notes", ""]
    for i, q in enumerate(report["contract_notes"], 1):
        lines.append(f"{i}. {q}")
    lines += ["", "## Training acceptance", "", report["training_acceptance"], ""]

    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source_root.expanduser().exists():
        print(f"source-root does not exist: {args.source_root}", file=sys.stderr)
        return 2
    inspect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

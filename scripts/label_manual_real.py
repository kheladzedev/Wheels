"""One-shot labeler for the 8 manual_real seed images.

Writes copies with safe stems into data/incoming/android_plugin_real/images/
and per-image annotation JSONs into annotations/. Coordinates are entered
once below as (normalized_x, normalized_y) pairs to insulate the script
from each image's resolution. The script multiplies by (W, H) at write
time so 5-10% eyeball error stays proportional across sources.

Annotation semantics here match what wheel_v3 was trained on (legacy rim
geometry) — A=left rim edge inside the wheel, B=right rim edge inside,
C=lowest visible point of the metal disc. This is intentional: mixing
floor-ray semantics from the 2026-05-13 contract with legacy rim
training data would give the model conflicting keypoint signals on a
tiny seed set. Re-labeling under the new contract is a separate /goal.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data" / "manual_real" / "images"
DST = REPO / "data" / "incoming" / "android_plugin_real"
DST_IMG = DST / "images"
DST_ANN = DST / "annotations"
DST_META = DST / "metadata"


# (source filename, target stem, [wheel_dicts])
# Each wheel_dict is in normalized image coords (0..1):
#   bbox = (x1, y1, x2, y2)
#   a, b, c = (x, y)
ENTRIES: list[tuple[str, str, list[dict]]] = [
    # 1. Lada Vesta SW Cross (2500x1250) — 3/4 front-left
    (
        "10_samyh_deshyovyh_mashin_v_Rossii_i_mire__1738651335.jpg",
        "real_001_lada_vesta",
        [
            {
                "bbox": (0.420, 0.580, 0.610, 0.890),
                "a": (0.450, 0.740),
                "b": (0.585, 0.740),
                "c": (0.515, 0.875),
            },
            {
                "bbox": (0.755, 0.635, 0.885, 0.880),
                "a": (0.775, 0.755),
                "b": (0.870, 0.755),
                "c": (0.825, 0.860),
            },
        ],
    ),
    # 2. BMW silver 5GT (1680x1115) — pure side view
    (
        "1644246291_1-hdpic-club-p-avtomobil-sboku-2.png",
        "real_002_bmw_5gt",
        [
            {
                "bbox": (0.080, 0.490, 0.250, 0.745),
                "a": (0.105, 0.620),
                "b": (0.230, 0.620),
                "c": (0.165, 0.720),
            },
            {
                "bbox": (0.685, 0.490, 0.855, 0.745),
                "a": (0.705, 0.620),
                "b": (0.830, 0.620),
                "c": (0.770, 0.720),
            },
        ],
    ),
    # 3. Mercedes GLA (1536x1024) — 3/4 front-right, only front-right wheel
    #    fully visible; rear wheel hidden behind body → omitted.
    (
        "cc7b4f9fc16e8192047257514d86572e_large.jpg",
        "real_003_mercedes_gla",
        [
            {
                "bbox": (0.530, 0.500, 0.795, 0.895),
                "a": (0.565, 0.700),
                "b": (0.765, 0.700),
                "c": (0.660, 0.860),
            },
        ],
    ),
    # 4. Nissan GT-R illustration (900x480) — pure side view
    (
        "foni-papik-pro-y8kb-p-kartinki-mashina-sboku-na-prozrachnom-fone-19.png",
        "real_004_gtr_render",
        [
            {
                "bbox": (0.125, 0.430, 0.300, 0.815),
                "a": (0.150, 0.610),
                "b": (0.280, 0.610),
                "c": (0.215, 0.790),
            },
            {
                "bbox": (0.690, 0.430, 0.870, 0.815),
                "a": (0.715, 0.610),
                "b": (0.850, 0.610),
                "c": (0.780, 0.790),
            },
        ],
    ),
    # 5. BMW X6 dark (284x177) — pure side view
    (
        "images (1).jpeg",
        "real_005_bmw_x6",
        [
            {
                "bbox": (0.180, 0.620, 0.345, 0.840),
                "a": (0.210, 0.720),
                "b": (0.320, 0.720),
                "c": (0.260, 0.815),
            },
            {
                "bbox": (0.665, 0.620, 0.825, 0.840),
                "a": (0.690, 0.720),
                "b": (0.800, 0.720),
                "c": (0.740, 0.815),
            },
        ],
    ),
    # 6. Mercedes GLC dealership (259x194) — 3/4 front-left
    #    Rear wheel mostly hidden → only front-left labeled.
    (
        "images (2).jpeg",
        "real_006_mercedes_glc",
        [
            {
                "bbox": (0.155, 0.610, 0.430, 0.910),
                "a": (0.200, 0.760),
                "b": (0.400, 0.760),
                "c": (0.290, 0.880),
            },
        ],
    ),
    # 7. Mini Cooper red (276x183) — pure side view
    (
        "images.jpeg",
        "real_007_mini_cooper",
        [
            {
                "bbox": (0.115, 0.560, 0.320, 0.900),
                "a": (0.155, 0.720),
                "b": (0.295, 0.720),
                "c": (0.220, 0.870),
            },
            {
                "bbox": (0.700, 0.560, 0.910, 0.900),
                "a": (0.730, 0.720),
                "b": (0.880, 0.720),
                "c": (0.800, 0.870),
            },
        ],
    ),
    # 8. Retro sedan (3840x2160) — pure side view, gold rims
    (
        "mashina_retro_vid_sboku_168158_3840x2160.jpg",
        "real_008_retro_sedan",
        [
            {
                "bbox": (0.175, 0.595, 0.325, 0.815),
                "a": (0.205, 0.700),
                "b": (0.305, 0.700),
                "c": (0.250, 0.790),
            },
            {
                "bbox": (0.710, 0.595, 0.860, 0.815),
                "a": (0.735, 0.700),
                "b": (0.835, 0.700),
                "c": (0.785, 0.790),
            },
        ],
    ),
]


def scale_pt(p: tuple[float, float], w: int, h: int) -> list[float]:
    return [round(p[0] * w, 1), round(p[1] * h, 1)]


def scale_bbox(b: tuple[float, float, float, float], w: int, h: int) -> list[float]:
    return [
        round(b[0] * w, 1),
        round(b[1] * h, 1),
        round(b[2] * w, 1),
        round(b[3] * h, 1),
    ]


def main() -> None:
    DST_IMG.mkdir(parents=True, exist_ok=True)
    DST_ANN.mkdir(parents=True, exist_ok=True)
    DST_META.mkdir(parents=True, exist_ok=True)

    for src_name, stem, wheels in ENTRIES:
        src_path = SRC_DIR / src_name
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        img = cv2.imread(str(src_path))
        if img is None:
            raise RuntimeError(f"cv2 cannot read {src_path}")
        h, w = img.shape[:2]
        ext = src_path.suffix.lower()
        dst_img = DST_IMG / f"{stem}{ext}"
        shutil.copyfile(src_path, dst_img)
        payload = {
            "frame_id": stem,
            "image": dst_img.name,
            "wheels": [
                {
                    "bbox_xyxy": scale_bbox(wd["bbox"], w, h),
                    "points": {
                        "a": scale_pt(wd["a"], w, h),
                        "b": scale_pt(wd["b"], w, h),
                        "c_disc_bottom": scale_pt(wd["c"], w, h),
                    },
                }
                for wd in wheels
            ],
        }
        (DST_ANN / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {dst_img.name}: {w}x{h}, {len(wheels)} wheel(s)")

    (DST_META / "source_info.json").write_text(
        json.dumps(
            {
                "source_name": "manual_real_seed_v1",
                "captured_with": "various web sources, hand-labeled",
                "captured_at": "2026-05-14",
                "image_count": len(ENTRIES),
                "notes": (
                    "Seed batch labeled visually for the wheel_v3 -> wheel_v4_real "
                    "fine-tune. A/B follow legacy rim semantics, not the "
                    "2026-05-13 floor-ray contract."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(ENTRIES)} entries into {DST}")


if __name__ == "__main__":
    main()

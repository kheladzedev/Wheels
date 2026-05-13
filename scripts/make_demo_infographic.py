"""Compose a single-page infographic of the pipeline output for sharing.

Reads a real inference image + its AR JSON and renders:
  - title + legend (real keypoint names, the contract)
  - the inference image (no fake data)
  - a compact JSON snippet matching what infer_image.py actually emits
  - a "what you see" panel

Usage:
    python scripts/make_demo_infographic.py \
        --image outputs/demo/manual_sample__sample_0019_final_pred.jpg \
        --json outputs/demo/manual_sample__sample_0019.json \
        --out outputs/demo/infographic.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def build_json_snippet(payload: dict) -> str:
    """Trim the payload to one wheel for readability while keeping the schema."""
    wheels = payload.get("wheels", [])
    snippet = {
        "frame_id": payload.get("frame_id", "frame_001"),
        "timestamp": round(payload.get("timestamp", 0.0), 1) or 1736700000.0,
        "wheels": [],
    }
    if wheels:
        w = wheels[0]
        snippet["wheels"].append(
            {
                "wheel_bbox": [round(v, 1) for v in w["wheel_bbox"]],
                "keypoints": [
                    {
                        "name": kp["name"],
                        "xy": [round(v, 1) for v in kp["xy"]],
                        "visibility": kp["visibility"],
                        "confidence": round(kp["confidence"], 2),
                    }
                    for kp in w["keypoints"]
                ],
                "confidence": round(w["confidence"], 2),
                "warnings": w.get("warnings", []),
            }
        )
        snippet["wheels"].append({"...": "(остальные колёса с такой же структурой)"})
    snippet["stats"] = {"n_wheels": len(wheels)}
    return json.dumps(snippet, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    img = Image.open(args.image)
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    json_text = build_json_snippet(payload)

    fig = plt.figure(figsize=(14, 11), facecolor="white")
    gs = GridSpec(
        3,
        2,
        height_ratios=[0.9, 3.0, 3.0],
        width_ratios=[2.0, 1.0],
        hspace=0.25,
        wspace=0.15,
        left=0.04,
        right=0.97,
        top=0.96,
        bottom=0.04,
    )

    # ---- Header (title + legend) ---------------------------------------
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_head.text(
        0.0, 0.85, "Sanity check на синтетике", fontsize=22, fontweight="bold", va="top"
    )
    ax_head.text(
        0.0,
        0.45,
        "Пример работы пайплайна: детекция колёс + 3 keypoints "
        "(rim_left, rim_right, disc_bottom).",
        fontsize=11,
        va="top",
        color="#444",
    )

    legend_items = [
        ("rim_left (верх обода)", (0, 1, 0), "o"),
        ("rim_right (низ обода)", (1, 0.78, 0), "o"),
        ("disc_bottom (низ диска)", (1, 0, 0), "o"),
        ("wheel bbox (колесо)", (1, 0.5, 0), "s"),
    ]
    base_x = 0.42
    for i, (label, color, marker) in enumerate(legend_items):
        col = i // 2
        row = i % 2
        x = base_x + col * 0.30
        y = 0.85 - row * 0.42
        if marker == "o":
            ax_head.scatter(
                [x],
                [y],
                s=160,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
        else:
            ax_head.add_patch(
                Rectangle(
                    (x - 0.012, y - 0.10),
                    0.024,
                    0.20,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=2,
                    transform=ax_head.transAxes,
                )
            )
        ax_head.text(x + 0.022, y, label, fontsize=10, va="center")
    ax_head.set_xlim(0, 1)
    ax_head.set_ylim(0, 1)

    # ---- Model output image -------------------------------------------
    ax_img = fig.add_subplot(gs[1, :])
    ax_img.imshow(img)
    ax_img.set_title(
        "Результат модели (синтетика)",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )
    ax_img.axis("off")

    # ---- JSON snippet -------------------------------------------------
    ax_json = fig.add_subplot(gs[2, 0])
    ax_json.axis("off")
    ax_json.set_title(
        "Пример JSON на выходе", loc="left", fontsize=14, fontweight="bold", pad=8
    )
    ax_json.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_json.transAxes,
            facecolor="#1e1e1e",
            edgecolor="none",
        )
    )
    ax_json.text(
        0.02,
        0.97,
        json_text,
        fontsize=8.5,
        family="monospace",
        color="#dcdcdc",
        va="top",
        ha="left",
        transform=ax_json.transAxes,
    )

    # ---- "What you see" panel ----------------------------------------
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis("off")
    ax_info.set_title("Что видно", loc="left", fontsize=14, fontweight="bold", pad=8)
    ax_info.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_info.transAxes,
            facecolor="#f5f5f5",
            edgecolor="#d0d0d0",
            linewidth=1,
        )
    )

    bullets = [
        ("✓", "Детектим колёса (wheel bbox)"),
        ("✓", "Находим 3 ключевые точки на каждом колесе:"),
        ("•", "    rim_left — верх обода"),
        ("•", "    rim_right — низ обода"),
        ("•", "    disc_bottom — низ металлического диска"),
        ("✓", "Отдаём confidence по колесу"),
        ("✓", "И по каждой точке отдельно"),
        ("✓", "frame_id + timestamp для матчинга\n   с камерой на стороне AR"),
    ]
    y = 0.93
    for mark, text in bullets:
        ax_info.text(
            0.05,
            y,
            mark,
            fontsize=11,
            va="top",
            fontweight="bold",
            color="#1a7f1a" if mark == "✓" else "#666",
            transform=ax_info.transAxes,
        )
        ax_info.text(0.13, y, text, fontsize=9.5, va="top", transform=ax_info.transAxes)
        y -= 0.08

    ax_info.text(
        0.05,
        0.20,
        "Важно",
        fontsize=11,
        fontweight="bold",
        color="#b00",
        transform=ax_info.transAxes,
    )
    ax_info.text(
        0.05,
        0.15,
        "Это синтетика, не реальная модель.\n"
        "Цель — проверить, что пайплайн\n"
        "работает end-to-end.",
        fontsize=9,
        color="#555",
        va="top",
        transform=ax_info.transAxes,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

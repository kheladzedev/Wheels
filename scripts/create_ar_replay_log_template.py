"""Write a fill-in AR replay JSONL template.

The output is intentionally named `*.template.jsonl` so it cannot be
mistaken for the real production-gate replay input. Replace the
FILL_ME fields with values from an actual AR device session before
running `src/validate_ar_replay.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("outputs/production_audit/ar_3d_replay.template.jsonl")


def build_observation(
    index: int,
    *,
    session_id: str,
    capture_device: str,
    capture_app_version: str,
    capture_date_utc: str,
    source_type: str,
    camera_pose_ref_prefix: str,
) -> dict[str, Any]:
    x0 = 612.4 + index * 0.35
    x1 = 861.7 + index * 0.2
    z = -2.05 - index * 0.001
    residual = 0.003 + (index % 5) * 0.001
    return {
        "schema_version": 1,
        "source_type": source_type,
        "capture_device": capture_device,
        "capture_app_version": capture_app_version,
        "capture_date_utc": capture_date_utc,
        "session_id": session_id,
        "frame_id": f"frame_{index:04d}",
        "capture_index": index,
        "camera_transform": None,
        "camera_pose_ref": f"{camera_pose_ref_prefix}_{index:04d}",
        "screen_points": {
            "a": [round(x0, 3), 742.1],
            "b": [round(x1, 3), 743.0],
            "c_disc_bottom": [round((x0 + x1) / 2.0, 3), 690.5],
        },
        "floor_raycast_hits": {
            "a": [1.18, 0.0, round(z, 4)],
            "b": [1.51, 0.0, round(z - 0.01, 4)],
        },
        "inlier": True,
        "residual": round(residual, 6),
        "recovered_plane": {
            "normal": [0.998, 0.0, 0.062],
            "point": [1.34, 0.0, round(z - 0.005, 4)],
            "support": 30,
        },
        "c_plane_hit": [1.34, 0.41, round(z - 0.005, 4)],
        "c_height_value": 0.41,
        "final_disc_bottom_position": [1.34, 0.41, round(z - 0.005, 4)],
    }


def build_template(
    *,
    observations: int,
    session_id: str,
    capture_device: str,
    capture_app_version: str = "FILL_ME_capture_app_version",
    capture_date_utc: str = "FILL_ME_YYYY-MM-DD",
    source_type: str,
    camera_pose_ref_prefix: str = "FILL_ME_pose",
) -> list[dict[str, Any]]:
    return [
        build_observation(
            index,
            session_id=session_id,
            capture_device=capture_device,
            capture_app_version=capture_app_version,
            capture_date_utc=capture_date_utc,
            source_type=source_type,
            camera_pose_ref_prefix=camera_pose_ref_prefix,
        )
        for index in range(observations)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--observations", type=int, default=30)
    parser.add_argument("--session-id", default="FILL_ME_session_id")
    parser.add_argument("--capture-device", default="FILL_ME_capture_device")
    parser.add_argument("--capture-app-version", default="FILL_ME_capture_app_version")
    parser.add_argument("--capture-date-utc", default="FILL_ME_YYYY-MM-DD")
    parser.add_argument(
        "--camera-pose-ref-prefix",
        default="FILL_ME_pose",
        help="Use a real camera-pose-store prefix for production logs, or leave FILL_ME in templates.",
    )
    parser.add_argument(
        "--source-type",
        default="FILL_ME_android_ar_device_replay",
        help="Use android_ar_device_replay/ios_ar_device_replay/ar_device_replay for real production logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations = build_template(
        observations=args.observations,
        session_id=args.session_id,
        capture_device=args.capture_device,
        capture_app_version=args.capture_app_version,
        capture_date_utc=args.capture_date_utc,
        source_type=args.source_type,
        camera_pose_ref_prefix=args.camera_pose_ref_prefix,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(obs, ensure_ascii=False) for obs in observations) + "\n",
        encoding="utf-8",
    )
    print(f"template={args.out} observations={len(observations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

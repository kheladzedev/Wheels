"""Sketchfab cars fetcher — discover + download free-downloadable car GLBs.

Searches the Sketchfab v3 API for downloadable car models, fetches each
GLB into an output directory, and writes a manifest mapping local file →
source UID + license + author. The GLB files are imported into UE via
the editor (drag-drop or python `import_asset` — out of scope here).

Token resolution order:
    1. `--token` CLI arg
    2. `SKETCHFAB_API_TOKEN` env var
    3. `.env` next to the repo root (single `KEY=VALUE` per line)

Usage:
    python src/fetch_sketchfab_cars.py \
        --output-dir data/sketchfab_cars \
        --max 50 \
        --query "car"

    python src/fetch_sketchfab_cars.py \
        --output-dir data/sketchfab_cars \
        --query "car" --query "pickup truck" --query "suv" \
        --max 24 --target-total 40

Network failures, 4xx rate limits, and license mismatches are logged and
skipped — never raised. Re-running with the same `--output-dir` skips
models that already have a `<uid>.glb` next to a `<uid>.json` manifest.

Dependencies: stdlib only (`urllib`, `json`, `argparse`). No `requests`
to stay inside the locked VSBL dep surface.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.sketchfab.com/v3"
EXIT_OK = 0
EXIT_TEMPORARY_BLOCK = 75
EXIT_CONSECUTIVE_FAILURES = 76
DEFAULT_QUERY = "car"
DEFAULT_OUTPUT_DIR = Path("data/sketchfab_cars")
DEFAULT_MAX = 50
DEFAULT_MAX_BYTES = 250 * 1024 * 1024
REQUEST_TIMEOUT_S = 60
PAGE_SIZE = 24
DOWNLOAD_FAILURE_KEY = "_download_failure"
LICENSE_ALLOWED = {
    "by",  # CC-BY
    "by-sa",  # CC-BY-SA
    "cc0",  # public domain
    "by-nd",  # CC-BY-ND (no derivatives — but ok for our render dataset)
}

VEHICLE_NAME_TERMS = {
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "auto",
    "automobile",
    "sedan",
    "suv",
    "coupe",
    "hatchback",
    "wagon",
    "pickup",
    "truck",
    "van",
    "bus",
    "motorhome",
    "roadster",
    "limousine",
    "taxi",
    "police",
    "rally",
    "race",
    "racing",
    "drift",
    "wheel",
    "wheels",
    "tire",
    "tyre",
    "rim",
    "hovercar",
    "porsche",
    "bmw",
    "toyota",
    "mazda",
    "nissan",
    "datsun",
    "fiat",
    "chevrolet",
    "ford",
    "gmc",
    "lamborghini",
    "ferrari",
    "audi",
    "mercedes",
    "volkswagen",
    "volvo",
    "honda",
    "hyundai",
    "kia",
    "lexus",
    "jeep",
    "tesla",
    "mustang",
    "corvette",
    "camaro",
    "countach",
}

CAR_BODY_NAME_TERMS = VEHICLE_NAME_TERMS - {
    "wheel",
    "wheels",
    "tire",
    "tyre",
    "rim",
}

NON_VEHICLE_NAME_TERMS = {
    "building",
    "house",
    "city",
    "street",
    "girl",
    "boy",
    "character",
    "cartoon",
    "asset set",
    "environment",
    "furniture",
    "chair",
    "table",
    "weapon",
}

NON_CAR_BODY_NAME_TERMS = {
    "bus stop",
    "driver",
    "drone",
    "hand truck",
    "lightbar",
    "nameplate",
    "rim",
    "siren",
    "tire",
    "tyre",
    "wheel",
    "wheels",
}


class SketchfabTemporaryBlock(RuntimeError):
    """Sketchfab is temporarily refusing API calls; stop instead of hammering."""


def _resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    env_token = os.environ.get("SKETCHFAB_API_TOKEN")
    if env_token:
        return env_token
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "SKETCHFAB_API_TOKEN":
                return value.strip().strip('"').strip("'")
    raise SystemExit("SKETCHFAB_API_TOKEN not found (CLI, env, or .env)")


def _api_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _search_models(
    token: str, query: str, max_count: int, *, strict_license: bool
) -> list[dict]:
    """Return up to max_count downloadable models matching the query.

    The /search endpoint does not always include license in the result
    summary, so by default we trust `downloadable=true` (Sketchfab only
    sets this on models the API consumer is permitted to download).
    `strict_license` re-applies the LICENSE_ALLOWED filter when the
    summary does include a slug.
    """
    out: list[dict] = []
    cursor = "0"
    while len(out) < max_count:
        params = urllib.parse.urlencode(
            {
                "type": "models",
                "q": query,
                "downloadable": "true",
                "archives_flavours": "false",
                "count": min(PAGE_SIZE, max_count - len(out)),
                "cursor": cursor,
                "sort_by": "-likeCount",
            }
        )
        url = f"{API_BASE}/search?{params}"
        try:
            page = _api_get(url, token)
        except urllib.error.HTTPError as exc:
            if exc.code == 405:
                raise SketchfabTemporaryBlock(
                    "Sketchfab search endpoint returned HTTP 405; "
                    "stop and retry later."
                ) from exc
            print(f"[search] HTTP {exc.code} — stopping pagination", file=sys.stderr)
            break
        results = page.get("results") or []
        if not results:
            break
        for item in results:
            if strict_license:
                license_ = (item.get("license") or {}).get("slug", "")
                if license_ and license_.lower() not in LICENSE_ALLOWED:
                    continue
            out.append(item)
            if len(out) >= max_count:
                break
        next_url = page.get("next")
        if not next_url:
            break
        cursor_q = urllib.parse.urlparse(next_url).query
        cursor_val = urllib.parse.parse_qs(cursor_q).get("cursor", [None])[0]
        if not cursor_val:
            break
        cursor = cursor_val
        time.sleep(0.3)
    return out


def _norm_name(value: str | None) -> str:
    return (value or "").lower().replace("_", " ").replace("-", " ")


def _name_tokens(name: str) -> set[str]:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name)
    return {tok for tok in cleaned.split() if tok}


def _contains_term(name: str, tokens: set[str], term: str) -> bool:
    if " " in term:
        return term in name
    return term in tokens


def _is_vehicle_like(model_or_manifest: dict) -> bool:
    name = _norm_name(model_or_manifest.get("name"))
    if not name:
        return False
    tokens = _name_tokens(name)
    has_non_vehicle = any(
        _contains_term(name, tokens, term) for term in NON_VEHICLE_NAME_TERMS
    )
    if has_non_vehicle:
        return any(
            _contains_term(name, tokens, term)
            for term in (
                "car",
                "truck",
                "vehicle",
                "bus",
                "taxi",
                "police",
                "wheel",
            )
        )
    return any(_contains_term(name, tokens, term) for term in VEHICLE_NAME_TERMS)


def _is_car_body_like(model_or_manifest: dict) -> bool:
    name = _norm_name(model_or_manifest.get("name"))
    if not name:
        return False
    tokens = _name_tokens(name)
    if any(_contains_term(name, tokens, term) for term in NON_CAR_BODY_NAME_TERMS):
        return False
    has_non_vehicle = any(
        _contains_term(name, tokens, term) for term in NON_VEHICLE_NAME_TERMS
    )
    if has_non_vehicle:
        return any(
            _contains_term(name, tokens, term)
            for term in ("car", "truck", "vehicle", "bus", "taxi")
        )
    return any(_contains_term(name, tokens, term) for term in CAR_BODY_NAME_TERMS)


def _matches_name_filter(model_or_manifest: dict, *, car_body_only: bool) -> bool:
    if car_body_only:
        return _is_car_body_like(model_or_manifest)
    return _is_vehicle_like(model_or_manifest)


def _count_value(model_or_manifest: dict, *keys: str) -> int | None:
    for key in keys:
        value = model_or_manifest.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _complexity_score(model_or_manifest: dict) -> tuple[int, int, str]:
    faces = _count_value(model_or_manifest, "face_count", "faceCount")
    vertices = _count_value(model_or_manifest, "vertex_count", "vertexCount")
    return (
        faces if faces is not None else 10**12,
        vertices if vertices is not None else 10**12,
        str(model_or_manifest.get("uid") or ""),
    )


def _passes_complexity_limits(
    model_or_manifest: dict,
    *,
    max_face_count: int | None,
    max_vertex_count: int | None,
) -> bool:
    faces = _count_value(model_or_manifest, "face_count", "faceCount")
    vertices = _count_value(model_or_manifest, "vertex_count", "vertexCount")
    if max_face_count is not None and faces is not None and faces > max_face_count:
        return False
    if max_vertex_count is not None and vertices is not None and vertices > max_vertex_count:
        return False
    return True


def _read_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(path: Path, model: dict) -> None:
    path.write_text(
        json.dumps(_manifest_for_model(model), indent=2, ensure_ascii=False)
    )


def _write_download_failure(path: Path, model: dict, reason: str) -> None:
    manifest = _manifest_for_model(model)
    existing = _read_manifest(path)
    merged = {**existing, **manifest}
    merged[DOWNLOAD_FAILURE_KEY] = {
        "reason": reason,
        "time": int(time.time()),
    }
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))


def _has_recent_download_failure(
    manifest: dict, *, skip_failed_within_hours: float, now: float | None = None
) -> bool:
    if skip_failed_within_hours <= 0:
        return False
    failure = manifest.get(DOWNLOAD_FAILURE_KEY)
    if not isinstance(failure, dict):
        return False
    try:
        failed_at = float(failure.get("time"))
    except (TypeError, ValueError):
        return False
    now = time.time() if now is None else now
    return now - failed_at < skip_failed_within_hours * 3600


def _count_existing_glbs(
    output_dir: Path, *, vehicle_only: bool, car_body_only: bool = False
) -> int:
    n = 0
    for glb in output_dir.glob("*.glb"):
        if not vehicle_only:
            n += 1
            continue
        if _matches_name_filter(
            _read_manifest(glb.with_suffix(".json")),
            car_body_only=car_body_only,
        ):
            n += 1
    return n


def _load_manifest_candidates(
    output_dir: Path,
    *,
    vehicle_only: bool,
    car_body_only: bool = False,
    skip_failed_within_hours: float = 0.0,
) -> list[dict]:
    out: list[dict] = []
    for manifest_path in sorted(output_dir.glob("*.json")):
        glb_path = manifest_path.with_suffix(".glb")
        if glb_path.exists():
            continue
        manifest = _read_manifest(manifest_path)
        uid = manifest.get("uid") or manifest_path.stem
        if not uid:
            continue
        manifest["uid"] = uid
        if vehicle_only and not _matches_name_filter(
            manifest,
            car_body_only=car_body_only,
        ):
            continue
        if _has_recent_download_failure(
            manifest,
            skip_failed_within_hours=skip_failed_within_hours,
        ):
            continue
        out.append(manifest)
    return out


def _move_rejected_existing(output_dir: Path, *, car_body_only: bool = False) -> int:
    rejected_dir = output_dir / "rejected"
    moved = 0
    for glb in list(output_dir.glob("*.glb")):
        manifest_path = glb.with_suffix(".json")
        manifest = _read_manifest(manifest_path)
        if _matches_name_filter(manifest, car_body_only=car_body_only):
            continue
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for p in (glb, manifest_path):
            if p.exists():
                p.replace(rejected_dir / p.name)
        moved += 1
    return moved


def _resolve_glb_url(
    token: str, uid: str, *, retries: int = 3, retry_sleep_s: float = 60.0
) -> str | None:
    """Return a temporary GLB download URL for a model UID, or None."""
    url = f"{API_BASE}/models/{uid}/download"
    for attempt in range(retries + 1):
        try:
            payload = _api_get(url, token)
            break
        except urllib.error.HTTPError as exc:
            print(f"[{uid}] download endpoint HTTP {exc.code}", file=sys.stderr)
            if exc.code == 405:
                raise SketchfabTemporaryBlock(
                    "Sketchfab download endpoint returned HTTP 405; "
                    "stop and retry later."
                ) from exc
            if exc.code == 429 and attempt < retries:
                time.sleep(retry_sleep_s)
                continue
            if exc.code == 429:
                raise SketchfabTemporaryBlock(
                    "Sketchfab download endpoint is rate-limited with HTTP 429; "
                    "stop and retry later."
                ) from exc
            return None
    glb = payload.get("glb")
    if isinstance(glb, dict) and "url" in glb:
        return glb["url"]
    gltf = payload.get("gltf")
    if isinstance(gltf, dict) and "url" in gltf:
        return gltf["url"]
    return None


def _download(
    url: str, out_path: Path, *, max_bytes: int, max_seconds: float | None
) -> bool:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as resp:
            length_header = resp.headers.get("Content-Length")
            if length_header is not None:
                size = int(length_header)
                if size > max_bytes:
                    print(
                        f"[download] {out_path.name} skipped: "
                        f"{size / 1024 / 1024:.1f} MB > "
                        f"{max_bytes / 1024 / 1024:.1f} MB",
                        file=sys.stderr,
                    )
                    return False
            total = 0
            started_at = time.monotonic()
            with tmp_path.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    if max_seconds is not None and time.monotonic() - started_at > max_seconds:
                        print(
                            f"[download] {out_path.name} aborted: "
                            f">{max_seconds:.0f}s download timeout",
                            file=sys.stderr,
                        )
                        tmp_path.unlink(missing_ok=True)
                        return False
                    total += len(chunk)
                    if total > max_bytes:
                        print(
                            f"[download] {out_path.name} aborted: "
                            f">{max_bytes / 1024 / 1024:.1f} MB",
                            file=sys.stderr,
                        )
                        tmp_path.unlink(missing_ok=True)
                        return False
                    fh.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"[download] {out_path.name} failed: {exc}", file=sys.stderr)
        return False
    if not tmp_path.exists():
        print(f"[download] {out_path.name} failed: temp file disappeared", file=sys.stderr)
        return False
    tmp_path.replace(out_path)
    return True


def _manifest_for_model(model: dict) -> dict:
    user = model.get("user")
    if isinstance(user, dict):
        user = user.get("username")
    return {
        "uid": model.get("uid"),
        "name": model.get("name"),
        "viewer_url": model.get("viewerUrl") or model.get("viewer_url"),
        "source_query": model.get("_source_query") or model.get("source_query"),
        "license": model.get("license"),
        "user": user,
        "face_count": model.get("faceCount") or model.get("face_count"),
        "vertex_count": model.get("vertexCount") or model.get("vertex_count"),
    }


def _download_model(
    token: str,
    model: dict,
    output_dir: Path,
    max_bytes: int,
    *,
    retries: int,
    retry_sleep_s: float,
    max_download_seconds: float | None,
) -> dict:
    uid = model.get("uid")
    name = model.get("name", "?")
    if not uid:
        return {"status": "failed", "uid": None, "name": name, "reason": "missing_uid"}

    glb_path = output_dir / f"{uid}.glb"
    manifest_path = output_dir / f"{uid}.json"
    _write_manifest(manifest_path, model)

    if glb_path.exists():
        return {"status": "existing", "uid": uid, "name": name}

    download_url = _resolve_glb_url(
        token, uid, retries=retries, retry_sleep_s=retry_sleep_s
    )
    if not download_url:
        _write_download_failure(manifest_path, model, "no_url")
        return {"status": "failed", "uid": uid, "name": name, "reason": "no_url"}
    ok = _download(
        download_url,
        glb_path,
        max_bytes=max_bytes,
        max_seconds=max_download_seconds,
    )
    if not ok:
        _write_download_failure(manifest_path, model, "download")
        return {"status": "failed", "uid": uid, "name": name, "reason": "download"}
    return {
        "status": "downloaded",
        "uid": uid,
        "name": name,
        "size_mb": glb_path.stat().st_size / 1024 / 1024,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--token", default=None, help="Sketchfab API token (overrides env/.env)"
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Search query string. Can be passed multiple times. "
            f"Default: {DEFAULT_QUERY!r}."
        ),
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX,
        help="Max search results to inspect per query",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help=(
            "Stop once output-dir contains this many .glb files. "
            "Existing files count toward the target."
        ),
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_BYTES / 1024 / 1024,
        help="Skip/abort any single model larger than this many MB.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download workers. Keep modest to avoid Sketchfab rate limits.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying Sketchfab HTTP 429 responses.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help=(
            "Number of retries for the per-model download endpoint. "
            "Set to 0 for short probe runs."
        ),
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.5,
        help="Delay between sequential download batches.",
    )
    parser.add_argument(
        "--max-download-seconds",
        type=float,
        default=180.0,
        help=(
            "Abort a single file download after this many seconds. "
            "Use 0 to disable the wall-clock timeout."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write <uid>.glb + <uid>.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only enumerate matches, no downloads",
    )
    parser.add_argument(
        "--strict-license",
        action="store_true",
        help="Drop models whose search summary reports a non-permissive license",
    )
    parser.add_argument(
        "--no-name-filter",
        action="store_true",
        help="Disable vehicle-like name filtering.",
    )
    parser.add_argument(
        "--car-body-only",
        action="store_true",
        help=(
            "Use a stricter name filter for whole car/vehicle body assets. "
            "This excludes standalone wheel/tire/rim models from candidate "
            "selection and target-total counting."
        ),
    )
    parser.add_argument(
        "--clean-rejected-existing",
        action="store_true",
        help="Move already-downloaded non-vehicle GLBs to output-dir/rejected.",
    )
    parser.add_argument(
        "--from-existing-manifests",
        action="store_true",
        help=(
            "Skip Sketchfab search and download from local JSON manifests "
            "whose matching GLB is absent."
        ),
    )
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=0,
        help=(
            "Skip this many candidate models after filtering. Useful for "
            "resuming/probing a large local manifest queue without always "
            "hitting the same UID first."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Only attempt this many candidate downloads in this run.",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Shuffle the filtered candidate queue before offset/limit slicing.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Deterministic seed used with --shuffle-candidates.",
    )
    parser.add_argument(
        "--sort-candidates",
        choices=("none", "small-first"),
        default="none",
        help="Sort filtered candidates before offset/limit slicing.",
    )
    parser.add_argument(
        "--max-face-count",
        type=int,
        default=None,
        help="Skip candidates whose manifest reports more than this many faces.",
    )
    parser.add_argument(
        "--max-vertex-count",
        type=int,
        default=None,
        help="Skip candidates whose manifest reports more than this many vertices.",
    )
    parser.add_argument(
        "--skip-failed-within-hours",
        type=float,
        default=0.0,
        help=(
            "When using --from-existing-manifests, skip JSON manifests whose "
            "last download attempt failed within this many hours. Use 0 to retry all."
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help=(
            "Stop the current run after this many sequential failed downloads. "
            "Use 0 to disable. Useful for unattended runs through cached manifests."
        ),
    )
    args = parser.parse_args(argv)

    token = _resolve_token(args.token)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_name_filter = not args.no_name_filter
    car_body_only = bool(args.car_body_only and use_name_filter)
    if args.clean_rejected_existing and use_name_filter:
        moved = _move_rejected_existing(args.output_dir, car_body_only=car_body_only)
        print(f"[clean] moved non-vehicle existing GLBs to rejected/: {moved}")

    max_bytes = int(args.max_mb * 1024 * 1024)
    max_download_seconds = (
        None if args.max_download_seconds <= 0 else float(args.max_download_seconds)
    )

    if args.from_existing_manifests:
        models_all = _load_manifest_candidates(
            args.output_dir,
            vehicle_only=use_name_filter,
            car_body_only=car_body_only,
            skip_failed_within_hours=args.skip_failed_within_hours,
        )
        print(f"[manifest] {len(models_all)} local manifests without GLBs")
    else:
        queries = args.query or [DEFAULT_QUERY]
        models_by_uid: dict[str, dict] = {}
        try:
            for query in queries:
                models = _search_models(
                    token, query, args.max, strict_license=args.strict_license
                )
                print(
                    f"[search] {len(models)} downloadable matches for '{query}' "
                    f"(before cross-query dedupe)"
                )
                for model in models:
                    uid = model.get("uid")
                    if not uid or uid in models_by_uid:
                        continue
                    if use_name_filter and not _matches_name_filter(
                        model,
                        car_body_only=car_body_only,
                    ):
                        continue
                    model["_source_query"] = query
                    models_by_uid[uid] = model
        except SketchfabTemporaryBlock as exc:
            print(f"[blocked] {exc}", file=sys.stderr)
            print(
                f"[done] downloaded=0 skipped_existing=0 skipped_failed=0 "
                f"skipped_target=0 glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=False)} "
                f"vehicle_glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=use_name_filter)} "
                f"car_body_glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=use_name_filter, car_body_only=True)}"
            )
            return EXIT_TEMPORARY_BLOCK

        models_all = list(models_by_uid.values())
        print(f"[search] {len(models_all)} unique downloadable matches total")

    downloaded = 0
    skipped_existing = 0
    skipped_failed = 0
    skipped_target = 0
    skipped_candidate_slice = 0
    candidates: list[dict] = []
    for model in models_all:
        uid = model.get("uid")
        if not uid:
            continue
        if (
            args.target_total is not None
            and _count_existing_glbs(
                args.output_dir,
                vehicle_only=use_name_filter,
                car_body_only=car_body_only,
            )
            >= args.target_total
        ):
            skipped_target += 1
            break
        glb_path = args.output_dir / f"{uid}.glb"
        if glb_path.exists():
            skipped_existing += 1
            continue
        if not _passes_complexity_limits(
            model,
            max_face_count=args.max_face_count,
            max_vertex_count=args.max_vertex_count,
        ):
            skipped_candidate_slice += 1
            continue
        candidates.append(model)

    if args.shuffle_candidates:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(candidates)

    if args.sort_candidates == "small-first":
        candidates.sort(key=_complexity_score)

    if args.candidate_offset:
        offset = max(0, args.candidate_offset)
        skipped_candidate_slice += min(offset, len(candidates))
        candidates = candidates[offset:]

    if args.candidate_limit is not None:
        limit = max(0, args.candidate_limit)
        skipped_candidate_slice += max(0, len(candidates) - limit)
        candidates = candidates[:limit]

    blocked = False
    stopped_for_failures = False
    if args.dry_run:
        print(f"[dry-run] candidates_to_download={len(candidates)}")
    else:
        workers = max(1, int(args.workers))
        batch_size = max(workers * 4, 8)
        consecutive_failures = 0
        for start in range(0, len(candidates), batch_size):
            if (
                args.target_total is not None
                and _count_existing_glbs(
                    args.output_dir,
                    vehicle_only=use_name_filter,
                    car_body_only=car_body_only,
                )
                >= args.target_total
            ):
                skipped_target += len(candidates) - start
                break
            batch = candidates[start : start + batch_size]
            if workers == 1:
                results = []
                for model in batch:
                    try:
                        result = _download_model(
                            token,
                            model,
                            args.output_dir,
                            max_bytes,
                            retries=args.download_retries,
                            retry_sleep_s=args.retry_sleep,
                            max_download_seconds=max_download_seconds,
                        )
                        results.append(result)
                        if result["status"] in {"downloaded", "existing"}:
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                        if (
                            args.max_consecutive_failures > 0
                            and consecutive_failures >= args.max_consecutive_failures
                        ):
                            print(
                                "[stopped] max consecutive download failures reached: "
                                f"{consecutive_failures}",
                                file=sys.stderr,
                            )
                            stopped_for_failures = True
                            break
                    except SketchfabTemporaryBlock as exc:
                        print(f"[blocked] {exc}", file=sys.stderr)
                        blocked = True
                        break
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    futures = [
                        executor.submit(
                            _download_model,
                            token,
                            model,
                            args.output_dir,
                            max_bytes,
                            retries=args.download_retries,
                            retry_sleep_s=args.retry_sleep,
                            max_download_seconds=max_download_seconds,
                        )
                        for model in batch
                    ]
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            results.append(future.result())
                        except SketchfabTemporaryBlock as exc:
                            print(f"[blocked] {exc}", file=sys.stderr)
                            blocked = True
                            break

            for result in results:
                status = result["status"]
                if status == "downloaded":
                    downloaded += 1
                    print(
                        f"[ok]  {result['uid']}  {result['size_mb']:6.1f} MB  "
                        f"{result['name'][:60]}"
                    )
                elif status == "existing":
                    skipped_existing += 1
                else:
                    skipped_failed += 1
            time.sleep(args.download_delay)
            if blocked or stopped_for_failures:
                break

    print(
        f"[done] downloaded={downloaded} "
        f"skipped_existing={skipped_existing} "
        f"skipped_failed={skipped_failed} "
        f"skipped_target={skipped_target} "
        f"skipped_candidate_slice={skipped_candidate_slice} "
        f"glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=False)} "
        f"vehicle_glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=use_name_filter)} "
        f"car_body_glbs_total={_count_existing_glbs(args.output_dir, vehicle_only=use_name_filter, car_body_only=True)}"
    )
    if blocked:
        return EXIT_TEMPORARY_BLOCK
    if stopped_for_failures:
        return EXIT_CONSECUTIVE_FAILURES
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

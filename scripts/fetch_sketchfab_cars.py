"""Fetch CC0-licensed downloadable car 3D scans from Sketchfab into
data/raw/sketchfab_cars/.

Sketchfab v3 API:
  GET https://api.sketchfab.com/v3/search?
      type=models &
      downloadable=true &
      license=cc0 &
      q=car

For each result, request its download URL:
  GET https://api.sketchfab.com/v3/models/{uid}/download

The response gives a presigned URL for the GLB or source archive. The
authenticated endpoints require either an API token (env SKETCHFAB_TOKEN)
or sketchfab cookie-based session. CC0 downloads are open but the URL
fetch still requires Authorization for the auth-stage to return the
presigned link.

If no token is configured, the script lists candidate uids only and exits
without downloading. The operator can then download manually and run
import_sketchfab_to_unreal.py against the saved files.

Usage:
    SKETCHFAB_TOKEN=xxx python scripts/fetch_sketchfab_cars.py \\
        --query "car scan" --limit 20 --out-dir data/raw/sketchfab_cars
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, parse, request


SKETCHFAB_API = "https://api.sketchfab.com/v3"


def _http_get(
    url: str, headers: dict[str, str] | None = None, timeout: int = 30
) -> bytes:
    req = request.Request(url, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except error.HTTPError as exc:
        sys.stderr.write(
            f"HTTP {exc.code} {url}: {exc.read().decode('utf-8', 'replace')[:200]}\n"
        )
        raise


def search_cc0_cars(
    query: str, limit: int, token: str | None, license_slug: str = "cc0"
) -> list[dict]:
    params = {
        "type": "models",
        "downloadable": "true",
        "license": license_slug,
        "q": query,
        "count": str(limit),
        "archives_flavours": "false",
    }
    url = f"{SKETCHFAB_API}/search?{parse.urlencode(params)}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    raw = _http_get(url, headers=headers)
    doc = json.loads(raw)
    return doc.get("results", [])


def get_download_link(uid: str, token: str) -> dict:
    url = f"{SKETCHFAB_API}/models/{uid}/download"
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    return json.loads(_http_get(url, headers=headers))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="car scan")
    p.add_argument(
        "--queries",
        default="",
        help="Comma-separated list of queries; overrides --query",
    )
    p.add_argument(
        "--licenses",
        default="cc0,cc-by,cc-by-sa",
        help="Comma-separated Sketchfab license slugs",
    )
    p.add_argument("--limit", type=int, default=20, help="Per query × license limit")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/sketchfab_cars"))
    p.add_argument("--token", default=os.environ.get("SKETCHFAB_TOKEN", ""))
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="List candidate uids+URLs only; do not download.",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sketchfab_manifest.json"

    queries = [q.strip() for q in args.queries.split(",") if q.strip()] or [args.query]
    licenses = [l.strip() for l in args.licenses.split(",") if l.strip()]
    print(
        f"searching: queries={queries} licenses={licenses} limit_per_pair={args.limit}",
        file=sys.stderr,
    )
    candidates: list[dict] = []
    seen_uids: set[str] = set()
    for q in queries:
        for lic in licenses:
            try:
                batch = search_cc0_cars(
                    q, args.limit, args.token or None, license_slug=lic
                )
            except Exception as exc:
                print(f"  search failed q={q!r} lic={lic}: {exc}", file=sys.stderr)
                continue
            new = 0
            for c in batch:
                uid = c.get("uid")
                if uid and uid not in seen_uids:
                    candidates.append(c)
                    seen_uids.add(uid)
                    new += 1
            print(
                f"  q={q!r} lic={lic}: +{new} (total {len(candidates)})",
                file=sys.stderr,
            )
            time.sleep(0.4)
    print(f"candidates total: {len(candidates)}", file=sys.stderr)

    manifest = []
    for c in candidates:
        entry = {
            "uid": c.get("uid"),
            "name": c.get("name"),
            "license": (c.get("license") or {}).get("slug", "?"),
            "viewer_url": c.get("viewerUrl"),
            "is_downloadable": c.get("isDownloadable", False),
            "tags": [t.get("name") for t in c.get("tags", [])][:8],
        }
        manifest.append(entry)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest_path}", file=sys.stderr)

    if args.manifest_only or not args.token:
        if not args.token:
            print(
                "WARNING: no SKETCHFAB_TOKEN set; skipping downloads. "
                "Edit manifest then download manually or rerun with token.",
                file=sys.stderr,
            )
        return

    # Download each model archive (Sketchfab returns presigned GLB or zip).
    downloaded = 0
    for entry in manifest:
        uid = entry["uid"]
        if not uid:
            continue
        try:
            dl = get_download_link(uid, args.token)
        except Exception as exc:
            entry["download_error"] = str(exc)
            continue
        # Prefer GLB if available, fall back to source archive.
        url = None
        for key in ("glb", "gltf", "source"):
            section = dl.get(key)
            if isinstance(section, dict) and section.get("url"):
                url = section["url"]
                entry["asset_format"] = key
                break
        if url is None:
            entry["download_error"] = "no_download_url"
            continue
        ext = ".glb" if entry.get("asset_format") in ("glb", "gltf") else ".zip"
        path = out_dir / f"{uid}{ext}"
        try:
            with request.urlopen(url, timeout=120) as resp:
                path.write_bytes(resp.read())
            entry["local_path"] = str(path)
            downloaded += 1
        except Exception as exc:
            entry["download_error"] = str(exc)
        time.sleep(0.5)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"downloaded: {downloaded}/{len(manifest)}", file=sys.stderr)


if __name__ == "__main__":
    main()

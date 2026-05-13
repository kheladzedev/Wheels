"""Scrape license-clean car photos from Wikimedia Commons.

Uses the MediaWiki search + imageinfo API to find File: pages matching
car-related keywords, filters by license (CC-BY / CC-BY-SA / CC0 /
public-domain), downloads a thumbnail of each into ``--images-dir``,
and appends provenance entries to ``--sources-json`` (one per image,
shape compatible with the existing ``manual_real/SOURCES.json``).

Stdlib only — no `requests`, no `bs4`. Wikimedia requires a contact
User-Agent (https://meta.wikimedia.org/wiki/User-Agent_policy); we set
one by default and let the CLI override.

Dedup: an image is skipped if its source URL OR its destination filename
already appears in ``--sources-json``. The script is resumable — re-run
with the same flags to top up.

Why search and not category traversal: category trees on Commons are
deep, uneven, and require manual seed selection. Free-text search via
``list=search&srnamespace=6`` is good enough for "wheels in the lower
third, 3/4 view, parked car" type queries and lets us diversify by
varying the query.

Usage:
    python src/fetch_wikimedia_cars.py \\
        --target-count 200 \\
        --images-dir   data/manual_real/images \\
        --sources-json data/manual_real/SOURCES.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "VSBL-WheelDataset/0.1 (https://github.com/local-vsbl; contact: edward_csm@tg)"
)

# Default search queries. Each one is hit independently and contributes a
# pool of candidate File: pages; the union is deduped before download.
# Phrased to bias towards photographs that match the AR use-case
# (parked car, ground-level view, wheels visible).
DEFAULT_QUERIES: tuple[str, ...] = (
    "parked car side view",
    "automobile three-quarter view",
    "car wheels close-up",
    "parking lot cars",
    "car side photograph",
    "vintage car side",
    "sedan parked",
    "hatchback exterior",
    "car alloy wheel",
    "city street parked cars",
)

# Free-licence substrings we accept. Anything else is skipped.
ACCEPTED_LICENSE_SUBSTRINGS: tuple[str, ...] = (
    "cc by",
    "cc-by",
    "cc0",
    "public domain",
    "publicdomain",
)

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MAX_FILENAME_LEN = 120
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Pure helpers — covered by tests/test_fetch_wikimedia_cars.py.
# ---------------------------------------------------------------------------


def slugify_title(title: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Wikimedia titles → filesystem-safe stems.

    ``File:Toyota Corolla 2010.jpg`` → ``Toyota_Corolla_2010``.
    """
    if title.lower().startswith("file:"):
        title = title[len("file:") :]
    stem = title.rsplit(".", 1)[0]
    stem = SAFE_NAME_RE.sub("_", stem).strip("_")
    if not stem:
        stem = "image"
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("_")
    return stem


def license_is_acceptable(license_str: str) -> bool:
    """``cc by 4.0`` / ``CC-BY-SA-3.0`` / ``CC0`` / ``Public domain`` → True."""
    if not license_str:
        return False
    needle = license_str.lower()
    return any(s in needle for s in ACCEPTED_LICENSE_SUBSTRINGS)


def extract_license(extmetadata: dict) -> str:
    """Pull the licence string out of MediaWiki's ``extmetadata`` blob.

    Tries ``LicenseShortName`` then ``License`` then ``UsageTerms``. The
    blob nests each field as ``{"value": "...", "source": "...", ...}``.
    """
    for key in ("LicenseShortName", "License", "UsageTerms"):
        entry = extmetadata.get(key)
        if isinstance(entry, dict):
            val = entry.get("value")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def derive_target_filename(title: str, index: int, prefix: str = "wmc") -> str:
    """Stable per-image filename: ``<prefix>_<index:04d>_<slug>.jpg``.

    Indexed so order is preserved across re-runs and existing files
    don't collide with new ones from the same title slug.
    """
    stem = slugify_title(title)
    return f"{prefix}_{index:04d}_{stem}.jpg"


def already_have(src_url: str, filename: str, existing_sources: Iterable[dict]) -> bool:
    """Skip if either the URL or the destination filename is already on disk."""
    src_url = src_url.strip()
    filename = filename.strip()
    for entry in existing_sources:
        if not isinstance(entry, dict):
            continue
        if entry.get("src_url", "").strip() == src_url:
            return True
        if entry.get("file", "").strip() == filename:
            return True
    return False


def load_sources_json(path: Path) -> list[dict]:
    """Read ``SOURCES.json`` if present, else return an empty list."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def save_sources_json(path: Path, entries: list[dict]) -> None:
    """Atomic-ish write of the merged sources list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(path)


def next_image_index(existing_sources: Iterable[dict], prefix: str = "wmc") -> int:
    """Highest existing ``wmc_NNNN_*`` index + 1, or 0 if none."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})_")
    max_idx = -1
    for entry in existing_sources:
        if not isinstance(entry, dict):
            continue
        m = pat.match(str(entry.get("file", "")))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# ---------------------------------------------------------------------------
# HTTP layer — kept narrow so it's easy to mock in tests.
# ---------------------------------------------------------------------------


def _http_get_json(url: str, user_agent: str, timeout: float = 30.0) -> dict:
    """GET ``url``, return parsed JSON. Raises on non-2xx."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _http_download(url: str, dest: Path, user_agent: str, timeout: float = 60.0) -> int:
    """Stream ``url`` into ``dest``. Returns the byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return total


def search_titles(
    query: str, *, limit: int, user_agent: str, sleep_s: float
) -> list[str]:
    """Wikimedia full-text search inside File namespace (=6)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",
        "srlimit": str(limit),
    }
    url = f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url, user_agent)
    hits = (data.get("query") or {}).get("search") or []
    titles: list[str] = []
    for h in hits:
        t = h.get("title")
        if isinstance(t, str) and t.lower().startswith("file:"):
            titles.append(t)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return titles


def fetch_imageinfo(
    titles: list[str], *, user_agent: str, thumb_width: int, sleep_s: float
) -> list[dict]:
    """One imageinfo call per chunk of titles. Returns parsed records."""
    out: list[dict] = []
    chunk_size = 25  # MediaWiki cap is 50, we keep headroom.
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i : i + chunk_size]
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": str(thumb_width),
            "titles": "|".join(chunk),
        }
        url = f"{API_ENDPOINT}?{urllib.parse.urlencode(params)}"
        data = _http_get_json(url, user_agent)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            iis = page.get("imageinfo")
            if not iis:
                continue
            ii = iis[0]
            out.append(
                {
                    "title": page.get("title"),
                    "thumb_url": ii.get("thumburl") or ii.get("url"),
                    "original_url": ii.get("url"),
                    "width": ii.get("thumbwidth") or ii.get("width"),
                    "height": ii.get("thumbheight") or ii.get("height"),
                    "mime": ii.get("mime"),
                    "extmetadata": ii.get("extmetadata") or {},
                }
            )
        if sleep_s > 0:
            time.sleep(sleep_s)
    return out


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape license-clean Wikimedia Commons car photos."
    )
    p.add_argument(
        "--target-count",
        type=int,
        default=170,
        help="Stop once this many new images have been downloaded.",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/manual_real/images"),
        help="Where to write the downloaded JPEGs.",
    )
    p.add_argument(
        "--sources-json",
        type=Path,
        default=Path("data/manual_real/SOURCES.json"),
        help="Provenance ledger (read for dedup, written for new entries).",
    )
    p.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=list(DEFAULT_QUERIES),
        help="Search queries. Defaults to a built-in mix.",
    )
    p.add_argument(
        "--per-query-limit",
        type=int,
        default=50,
        help="Max File: hits to request per query (Wikimedia hard cap is 500).",
    )
    p.add_argument(
        "--thumb-width",
        type=int,
        default=1280,
        help="Thumbnail width in pixels to download (server-side resize).",
    )
    p.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent. Wikimedia requires a contactable string.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API calls (be a good citizen).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and filter candidates, but don't download or write.",
    )
    return p.parse_args(argv)


def collect_candidate_records(
    queries: list[str],
    *,
    per_query_limit: int,
    user_agent: str,
    thumb_width: int,
    sleep_s: float,
) -> list[dict]:
    """For each query, search → imageinfo. Dedup by title across queries."""
    titles_seen: set[str] = set()
    titles_ordered: list[str] = []
    for q in queries:
        try:
            hits = search_titles(
                q, limit=per_query_limit, user_agent=user_agent, sleep_s=sleep_s
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"WARN: search failed for {q!r}: {e}", file=sys.stderr)
            continue
        for t in hits:
            if t not in titles_seen:
                titles_seen.add(t)
                titles_ordered.append(t)
        print(f"  query {q!r}: +{len(hits)} hits (total unique: {len(titles_ordered)})")

    if not titles_ordered:
        return []
    try:
        return fetch_imageinfo(
            titles_ordered,
            user_agent=user_agent,
            thumb_width=thumb_width,
            sleep_s=sleep_s,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"ERROR: imageinfo fetch failed: {e}", file=sys.stderr)
        return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    existing = load_sources_json(args.sources_json)
    print(f"Loaded {len(existing)} existing source entries from {args.sources_json}")

    records = collect_candidate_records(
        args.queries,
        per_query_limit=args.per_query_limit,
        user_agent=args.user_agent,
        thumb_width=args.thumb_width,
        sleep_s=args.sleep,
    )
    print(f"Candidate records from Wikimedia: {len(records)}")

    args.images_dir.mkdir(parents=True, exist_ok=True)
    new_entries: list[dict] = []
    seen_filenames: set[str] = {
        str(e.get("file", "")) for e in existing if isinstance(e, dict)
    }
    next_idx = next_image_index(existing)
    downloaded = 0

    for rec in records:
        if downloaded >= args.target_count:
            break

        title = rec.get("title") or ""
        thumb_url = rec.get("thumb_url") or ""
        mime = rec.get("mime") or ""
        if not (title and thumb_url):
            continue
        if not mime.startswith("image/"):
            continue

        licence = extract_license(rec.get("extmetadata") or {})
        if not license_is_acceptable(licence):
            continue

        filename = derive_target_filename(title, next_idx)
        while filename in seen_filenames:
            next_idx += 1
            filename = derive_target_filename(title, next_idx)

        if already_have(thumb_url, filename, existing + new_entries):
            continue

        dest = args.images_dir / filename
        if args.dry_run:
            print(f"DRY-RUN would download: {title} -> {filename} ({licence})")
            new_entries.append(
                {
                    "file": filename,
                    "src_title": title,
                    "src_url": thumb_url,
                    "license": licence,
                    "bytes": 0,
                    "shape_hw": [
                        int(rec.get("height") or 0),
                        int(rec.get("width") or 0),
                    ],
                }
            )
            seen_filenames.add(filename)
            next_idx += 1
            downloaded += 1
            continue

        try:
            n_bytes = _http_download(thumb_url, dest, args.user_agent)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"WARN: download failed for {title}: {e}", file=sys.stderr)
            continue

        new_entries.append(
            {
                "file": filename,
                "src_title": title,
                "src_url": thumb_url,
                "license": licence,
                "bytes": n_bytes,
                "shape_hw": [int(rec.get("height") or 0), int(rec.get("width") or 0)],
            }
        )
        seen_filenames.add(filename)
        next_idx += 1
        downloaded += 1
        print(
            f"  [{downloaded}/{args.target_count}] {filename} "
            f"({n_bytes // 1024} KiB, {licence})"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    if not args.dry_run and new_entries:
        save_sources_json(args.sources_json, existing + new_entries)

    print()
    print("Fetch summary:")
    print(f"  Candidates considered: {len(records)}")
    print(f"  New images downloaded: {downloaded}")
    print(f"  SOURCES.json entries:  {len(existing) + len(new_entries)}")
    print(f"  Images dir:            {args.images_dir}")
    print(f"  Sources file:          {args.sources_json}")
    if args.dry_run:
        print("  Mode:                  DRY-RUN (no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

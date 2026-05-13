---
description: Run VSBL repo healthcheck — pytest + plugin synthetic ingestion smoke (no GPU, no training).
argument-hint: (no arguments)
---

# /healthcheck — VSBL repo healthcheck

Run the four invariants the project relies on at any commit. Fast, no
GPU, no training. Mirrors `scripts/healthcheck.sh` so the slash command
and the shell entrypoint stay in sync.

If any step exits non-zero, **stop and report the failing command +
output verbatim**. Do not "fix forward" silently.

## Steps (run in order, in foreground, surface output)

### 1. Tests are green

```bash
./.venv/bin/pytest -q
```

Expected: exit `0`, summary line ends with `passed`. If failures, list
the failing test IDs and the first traceback line.

### 2. Plugin synthetic generator end-to-end

```bash
./.venv/bin/python src/create_sample_keypoint_incoming.py --count 20 --overwrite
```

Expected: writes 20 images + 20 annotations + `metadata/source_info.json`
under `data/incoming/android_plugin/`.

### 3. Incoming validator accepts its own synthetic output

```bash
./.venv/bin/python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin
```

Expected: `Errors: 0`. Warnings are OK; ERROR-level findings are not.

### 4. Previewer renders 5 incoming samples

```bash
./.venv/bin/python src/preview_keypoint_annotations.py --source-root data/incoming/android_plugin --count 5
```

Expected: 5 files under `outputs/keypoint_preview/`.

## Final report

After all four pass, report:

- `pytest`: N passed.
- Synthetic batch: `data/incoming/android_plugin/` regenerated, 20 images.
- Incoming check: clean.
- Preview: 5 files in `outputs/keypoint_preview/`.

End with one of: `Healthcheck OK.` or `Healthcheck FAILED at step N` +
the exact failing command.

## Notes

- This command does **not** run the YOLO-pose converter, the dataset
  checker, or the YOLO-pose previewer. For that, use
  `/plugin-ingestion-smoke`.
- This command does **not** touch training or inference paths.
- The synthetic generator overwrites `data/incoming/android_plugin/`. If
  a real batch is staged there, healthcheck will clobber it — back it
  up first.

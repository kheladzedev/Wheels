# Claude Code workflow for VSBL

How Claude is expected to operate in this repository. The repo-level
guardrails are in `CLAUDE.md`; this document is process-level (when to
plan, when to ask, how to report). Both apply at the same time.

## When to use `/goal`

`/goal` declares a binding done-condition; Claude cannot stop until the
condition is met. Use it for tasks where the wrong "done" is expensive:

- Adding or modifying a converter, validator, preview, or any code that
  feeds training data.
- Changing inference / postprocess / export — anything the AR client
  might consume.
- Multi-file refactors (more than ~3 files in flight at once).
- Setting up tooling, skills, hooks, or workflow infrastructure (this
  document was produced under a `/goal`).

Do **not** use `/goal` for:

- One-line bug fixes or typo corrections.
- Exploratory questions ("how does X work?").
- Read-only audits.

## How to define the done-condition

A done-condition is a checklist of **externally verifiable** facts. Aim
for the format the project already uses (see this task's spec, or
`/goal` invocations in `~/.claude/project-memory/VSBL.md`):

1. Numbered, machine-checkable items. "Created `src/foo.py`" is checkable;
   "improved code quality" is not.
2. Explicit commands that must pass. Paste them verbatim into the goal —
   Claude runs them as written.
3. A list of files / behaviours that must **not** change. Without this,
   refactor creep is easy.
4. A final-summary template: changed files, commands, test result,
   source of truth, next goal.

If the user gives an under-specified `/goal`, ask one clarifying question
**before** starting work, not in the middle.

## What checks Claude must run before claiming done

Always:

```bash
./.venv/bin/pytest -q
```

Plus the smoke chain for whichever pipeline you touched — see the
`ml-pipeline-review` skill, §2. If you touched the AR JSON shape, also
run `tests/test_ar_contract.py` explicitly and call it out in the
summary.

If a check fails, Claude does **not** stop with success. The failure
and its root cause go into the summary verbatim.

## What must not be changed without explicit confirmation

The non-negotiables, from `CLAUDE.md` → "Do not break":

- **Confirmed AR JSON schema** (`frame_id`, `wheels[].bbox_xyxy`,
  `confidence`, `points.{a, b, c_disc_bottom}`).
- **Legacy pipeline** (`convert_incoming_to_yolo.py` / `check_dataset.py`
  / `preview_labels.py` / `configs/dataset.yaml` / `data/wheel_dataset/`).
- **Plugin pipeline** (`convert_keypoint_incoming_to_yolo_pose.py` /
  `check_yolo_pose_dataset.py` / `preview_yolo_pose_labels.py` /
  `configs/pose_dataset.yaml` / `data/wheel_pose_dataset/`).
- **Training / inference code** (`train_yolo.py`, `infer_image.py`,
  `infer_batch.py`, `postprocess_wheels.py`, `export_model.py`) unless
  the goal explicitly names it.
- **Dependency surface** — stdlib, `opencv-python`, `numpy`, `pytest`,
  `ultralytics`. No torch outside ultralytics. No tracking libs.

If a task seems to require touching one of these and the goal didn't
authorise it: stop, surface the conflict, ask. Do not silently expand
scope.

## Tools Claude should reach for first

- **Pure-Python edits**: `Edit` / `Write`. Never shell `sed` / `awk`.
- **File inspection**: `Read`. Avoid `cat` / `head` via `Bash`.
- **Tests**: `./.venv/bin/pytest -q` from `Bash`.
- **Big or open-ended exploration**: spawn `Explore` (read-only) or
  `feature-dev:code-explorer` subagents to keep the main context lean.
- **Contract review**: skill `vsbl-ar-contract` BEFORE editing
  contract-bearing code.
- **Dataset review**: skill `yolo-pose-dataset` BEFORE editing
  converters/checkers/previewers.
- **Pre-done review**: skill `ml-pipeline-review` BEFORE writing the
  final summary.

## How to report the final summary

Match the format `ml-pipeline-review` §6 prescribes:

1. **Changed files** — group by purpose if helpful, but list every path.
2. **Commands run** — verbatim, in order. Don't paraphrase.
3. **Test result** — pytest count + status. If failures, list test IDs
   and root cause.
4. **What is now the source of truth** — which file/skill/doc the next
   contributor reads first.
5. **Open items / next recommended goal** — what's still blocked, what
   to do next.

Be terse. Repo style is no trailing summaries, no emojis. Prefer two
clear sentences to a paragraph.

## When to ask the user

Before:

- Adding a new dependency.
- Touching a "Do not break" item.
- Renaming or removing an exported function.
- Changing a JSON schema field name or shape.
- Long-running operations (full training, large data downloads).

After (status update):

- Each failing pytest, with the failure surface.
- Each step in a multi-step goal once it's done (TaskUpdate is the
  surface — keep it current).

Never ask for permission to:

- Read files.
- Run already-allow-listed commands (see `.claude/settings.json`).
- Run `pytest`, smoke scripts, healthcheck.

## See also

- Root `CLAUDE.md` — repo-level guardrails.
- Skills under `.claude/skills/` — `vsbl-ar-contract`,
  `yolo-pose-dataset`, `ml-pipeline-review`.
- `docs/AR_ML_CONTRACT.md` — the confirmed contract narrative.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — pending AR-team sign-offs.

# Spec Compliance — AR Mechanic "Примерка колес"

Maps each requirement from the AR-team specification document
("Механика «Примерка колес»", source PDF dated 2026-05-13) to the
concrete file, function, or open question in this repo.

Use this as the single entry point when verifying that an ML change
does not break the AR contract, or when onboarding a reviewer who has
only the PDF in hand.

## Source

- Title: **Механика «Примерка колес»**
- Sections covered: Подготовка / Старт / Использование / Стоп / Main /
  «Как разрабатывать» (Mock-система).
- This document tracks what the **ML side** is contractually obliged to
  deliver. Everything 3D (raycast / RANSAC / plane recovery / K-frame
  accumulation / cross-frame association) stays on the **AR side** per
  `docs/AR_ML_CONTRACT.md`.

## Lifecycle mapping

The spec describes a five-stage lifecycle. ML only participates in
stage 3 ("Использование"). The other four stages are AR-internal.

| Stage           | AR-side concern                                            | ML-side concern |
|-----------------|------------------------------------------------------------|---|
| Подготовка      | Find a floor plane over N m²                               | None |
| Старт           | Initialise NN, buffers, async pipeline, switch UI to search| Boot the YOLO-pose model; warm-up is `model.predict(...)` on a dummy frame if needed |
| Использование   | Per-frame loop: capture → save camera transform → invoke NN → accumulate → RANSAC → visualise | **All ML work**: per-frame, stateless, per-wheel 3-keypoint detection |
| Стоп            | Replace temporary markers with final 3D wheel planes       | None |
| Main            | Object selection, colour change, "Найти ещё" button        | None unless "Найти ещё" re-enters stage 3 |

## ML deliverable per the spec

The "Использование" stage names exactly four pieces of data that ML
must produce per frame:

1. **Detected wheels** as a list, possibly multiple per frame.
2. **Keypoints per wheel**, used by AR for:
   - Plane recovery via two screen-space ray sources (points A, B).
   - Disc height via one raycast onto the recovered plane (point C =
     `disc_bottom`).
3. **Frame identification** so AR can pair the response with the camera
   transform it saved at capture time.
4. **Wheel-level confidence** for coarse filtering/debug. Per-keypoint
   confidence and visibility are not part of the confirmed JSON; wheels
   with any occluded/missing A/B/C point are omitted entirely.

This is exactly the payload contract in `docs/AR_ML_CONTRACT.md`.

## Per-requirement coverage

Each row maps one PDF requirement to where it lives in the repo.

| Spec text (PDF) | ML obligation | Where implemented / documented |
|---|---|---|
| "Нейросеть должна поддерживать обнаружение нескольких колес в одном кадре" | Multi-instance detection per frame | `src/infer_image.py` (iterates every box); `src/postprocess_wheels.py` `build_ar_payload`; round-trip tested at `tests/test_postprocess_wheels.py` |
| "ключевые точки колеса" | 3 keypoints per wheel: `points.a`, `points.b`, `points.c_disc_bottom` | `docs/KEYPOINT_SPEC.md`; confirmed shape pinned by `tests/test_confirmed_ar_schema_shape.py` and `tests/test_ar_contract.py` |
| "список найденных колес (может быть несколько)" | `wheels` is a list (zero or more) | `build_ar_payload` returns `{"wheels": [...]}`; pinned by `test_ar_contract.py` |
| "Сохраняется трансформация камеры в момент захвата кадра" | AR owns transform; ML echoes `frame_id` only in confirmed JSON | `src/infer_image.py`; `src/infer_batch.py`; pinned by confirmed-schema tests. `timestamp` exists only in legacy/debug payloads |
| "Через некоторое время нейросеть возвращает" (async) | Stateless, per-frame inference; no temporal coupling | `docs/AR_ML_CONTRACT.md` → "ML is per-frame and stateless" |
| "Восстановление позиции … raycast в плоскость пола … положение и ориентация плоскости колеса" | NOT ML | `docs/AR_ML_CONTRACT.md` Out-of-scope section; `CLAUDE.md` "What NOT to do here: don't add 3D / RANSAC / plane fitting" |
| "Процесс повторяется K раз. Накапливаются наблюдения по каждому колесу" | NOT ML; K is AR-side budget | Open question `docs/QUESTIONS_FOR_TEAM.md` (K value not pinned) |
| "Применяется RANSAC … Удаляются шумы и выбросы … Формируется стабильная плоскость колеса" | NOT ML | `docs/AR_ML_CONTRACT.md` Out-of-scope; `CLAUDE.md` |
| "нижняя точка диска, полученная от нейросети" | Emit `points.c_disc_bottom` per wheel | `docs/KEYPOINT_SPEC.md`; `src/postprocess_wheels.py::to_confirmed_schema`; eval report key `c_disc_bottom` |
| "raycast на ранее восстановленную плоскость … значения усредняются" | NOT ML | AR-side disc-height accumulation, documented in `AR_ML_CONTRACT.md` |
| "временное обозначение … «кубик»" | NOT ML | AR visualisation only |
| "Стоп … нейросеть останавливается" | Inference loop is AR-driven; ML script is invoked per call | `src/infer_image.py` is one-shot per image; no persistent state to "stop" |
| "Main-состояние … выбрать колесо; менять цвет" | NOT ML | AR-side object manipulation |
| "Mock-система: три точки в центре экрана … raycast из двух экранных точек … раскраска по третьей" | NOT ML; confirms 3-keypoint contract | `docs/KEYPOINT_SPEC.md` "Why three points specifically" explicitly mirrors this section |
| "Чтобы корректно подобрать ransac параметры, мне нужен лог попаданий" | AR owns hit logs because they require camera transform + floor raycast hits; ML can provide per-frame 2D detections for those logs | `docs/AR_MOCK_LOG_CONTRACT.md`; `src/infer_batch.py`; `src/infer_image.py` |

## Negative invariants the spec implies for ML

These are things ML must NOT do. All are enforced by tests and policy.

- No 3D world positions, no plane equations, no RANSAC residuals,
  no inlier counts in the confirmed JSON. Tested in
  `tests/test_confirmed_ar_schema_shape.py` and `tests/test_ar_contract.py`.
- No cross-frame `track_id`. Same tests above.
- No `timestamp`, per-keypoint `visibility`, or per-keypoint confidence
  in the confirmed JSON.
- No temporal smoothing or per-wheel state. `infer_image.py` is one-shot.
- No camera intrinsics or extrinsics returned by ML. AR has these.

## Acceptance: every spec line maps somewhere

The table above covers every concrete obligation the PDF places on ML.
If a future spec revision adds a requirement, add a row here AND add a
contract test before changing the schema.

## Still-open items pending AR sign-off

These come from `docs/OPEN_QUESTIONS_AR_SPEC.md`. They are not contract
violations — they are clarifications the spec did not include.

- §9 Acceptable keypoint error budget. AR explicitly said raw 2D pixel
  budget is not enough; final acceptance should be 3D-side after raycast
  and RANSAC. `src/eval_keypoints.py` still reports per-point pixel error
  as an ML diagnostic.
- §7 Unreal export capabilities (does it include 3D positions of A/B/C?).

Status (post 2026-05-14 schema-drift fix):
- `src/infer_image.py` writes the **confirmed AR schema** as the primary
  `<stem>.json`; legacy intermediate is `<stem>_legacy.json`. The
  `--confirmed-schema` flag is a no-op alias.
- `src/infer_batch.py` writes the **confirmed AR schema** as the primary
  `<stem>__frame_XXX.json`; legacy intermediate is
  `<stem>__frame_XXX_legacy.json` when `--emit-legacy` is passed
  (`--target-schema` is a deprecated alias for the same). The historic
  pre-confirmed "target" draft schema is no longer emitted by either
  entry point.

## How to use this document

- **Before editing the JSON contract**: read this file plus
  `docs/AR_ML_CONTRACT.md`. If a row maps to a test, run it
  (`./.venv/bin/pytest tests/test_ar_contract.py -v`) before and after
  the change.
- **When AR team asks "do you handle X?"**: search this table for the
  spec line. The "Where implemented" column is the canonical answer.
- **When a new PDF revision lands**: diff the new PDF against this
  table. New rows go here first, then to contract tests, then to code.

## See also

- `docs/AR_ML_CONTRACT.md` — full responsibility split and target JSON.
- `docs/KEYPOINT_SPEC.md` — A/B/C geometric definitions.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — open AR-team confirmation items.
- `docs/QUESTIONS_FOR_TEAM.md` — broader open questions (K, N, error
  budget, platforms).
- `docs/TASK_PLAN.md` — staged delivery plan.
- `tests/test_ar_contract.py` — live contract guard.

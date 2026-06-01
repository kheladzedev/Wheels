# ML/AR Handoff Report — VSBL Wheel Fitting

> **CORRECTION 2026-05-30.** Live full test suite is **1031 passed, 0 failed**
> (not "495 passed, 1 skipped" cited below). Any PT-vs-export parity number is
> **raw-tensor parity**, not decoded-keypoint parity; strict decoded-keypoint
> parity is diagnostic/relaxed (see `docs/EXPORT_PARITY_AUDIT.md`). The shipped
> champion is the YOLO11s-pose model at
> `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`; the MN2
> track referenced in older sections is a proposed, untrained architecture.

**Audience:** AR / Unreal client team, project lead.
**Source repository:** VSBL repository
**Date of hand-off:** 2026-05-21
**Status of this document:** authoritative for ML→AR integration as of this date. Subsequent changes must be reflected in `docs/AR_ML_CONTRACT.md` first; this report is a snapshot.

---

## 1. Project status

ML-сторона VSBL готова к интеграции с AR-клиентом по контракту, подтверждённому AR-командой 2026-05-13 и re-confirmed 2026-05-18 (Unreal follow-up).

- **Schema, runtime guards, output paths, documentation:** READY.
- **Test suite:** 1031 passed, 0 failed, 2 warnings (deprecation от `torch.onnx`, не наше). (was: «495 passed, 1 skipped» — stale 2026-05-30.)
- **Model prediction quality:** PROVISIONAL. Текущий чекпойнт обучен на синтетике + provisional Unreal-bundle. Real-data validation ещё не завершена (см. §15).
- **Production hand-off:** возможен для technical-integration testing на стороне AR **сейчас**; production-quality predictions требуют пунктов §15 и §18.

---

## 2. Confirmed ML/AR contract

Source of truth: `docs/AR_ML_CONTRACT.md`.

Поддерживающие документы:

- `docs/KEYPOINT_SPEC.md` — геометрия A/B/C.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — resolved / still-open items.
- `tests/test_confirmed_ar_schema_shape.py`, `tests/test_ar_contract.py`, `tests/test_infer_batch_schema.py` — pinned shape; любое нарушение валится в CI.

Контрактные инварианты:

1. ML возвращает **per-frame** JSON в screen-space pixels.
2. Per frame — **массив колёс** (длина ≥ 0, multi-wheel поддерживается).
3. Per wheel — **3 keypoint'а** в фиксированном порядке `a`, `b`, `c_disc_bottom`.
4. `frame_id` — string, **echoed** AR-стороной (ML только возвращает обратно).
5. ML не возвращает 3D, никакой пост-обработки сцены, никакого tracking, никакого RANSAC.

Изменения контракта впредь — только через explicit AR sign-off, фиксируются в `docs/OPEN_QUESTIONS_AR_SPEC.md` и обновлением `tests/test_ar_contract.py`.

---

## 3. Exact JSON output example

Один кадр, два полностью видимых колеса:

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [120.5, 220.0, 280.5, 380.0],
      "confidence": 0.93,
      "points": {
        "a":             [135.0, 372.0],
        "b":             [265.0, 374.0],
        "c_disc_bottom": [205.0, 320.0]
      }
    },
    {
      "bbox_xyxy": [420.0, 230.0, 560.0, 370.0],
      "confidence": 0.88,
      "points": {
        "a":             [428.0, 362.0],
        "b":             [552.0, 363.0],
        "c_disc_bottom": [490.0, 318.0]
      }
    }
  ]
}
```

Пустой кадр (валидный):

```json
{ "frame_id": "frame_0042", "wheels": [] }
```

---

## 4. Field definitions

| Field | Type | Required | Format | Notes |
|---|---|---|---|---|
| `frame_id` | string | yes | non-empty | Echoed back from AR. Used to match the camera transform AR saved at capture time. |
| `wheels` | array | yes (may be `[]`) | length ≥ 0 | Zero or more detected wheels in this frame. Empty array is valid и означает «не обнаружено». |
| `wheels[].bbox_xyxy` | array[4] of float | yes | pixels, top-left origin, `x1 < x2`, `y1 < y2` | Top-left + bottom-right corners. Integers also valid; Python serialiser always emits float. |
| `wheels[].confidence` | float | yes | `[0, 1]` | Wheel-level detection confidence. **No per-keypoint confidence.** |
| `wheels[].points` | object | yes | closed set: `a`, `b`, `c_disc_bottom` | Других ключей не будет. |
| `wheels[].points.a` | array[2] of float | yes | screen-space pixels | См. §5. |
| `wheels[].points.b` | array[2] of float | yes | screen-space pixels | См. §5. |
| `wheels[].points.c_disc_bottom` | array[2] of float | yes | screen-space pixels | См. §5. |

---

## 5. Point semantics

### `points.a` — A, left floor-ray point

A/B are 2D screen-space pixels located on the visible floor/ground region near the wheel footprint. AR raycasts these pixels onto the detected floor plane.

Конкретно для A: 2D screen-space пиксель в зоне видимого пола слева от точки контакта шины с полом, рядом с основанием колеса. AR использует raycast этого пикселя в свою floor-плоскость, чтобы получить левый anchor вертикальной плоскости колеса.

**A — не пиксель на металлическом ободе, не на шине, не на самом колесе.** Это пиксель в зоне видимого пола около основания колеса.

### `points.b` — B, right floor-ray point

Симметрично A: 2D screen-space пиксель в зоне видимого пола справа от точки контакта шины с полом. AR raycast'ит в пол и получает правый anchor.

**B — не пиксель на металлическом ободе, не на шине, не на самом колесе.**

### `points.c_disc_bottom` — C, lowest visible disc-bottom point

2D screen-space пиксель в нижней видимой границе металлического обода / диска — в месте, где обод встречается с резиной снизу колеса. AR raycast'ит C **на уже восстановленную вертикальную плоскость колеса** (не в пол) и получает 3D-позицию нижней точки диска для расчёта высоты установки виртуального колеса.

**C — пиксель на видимой металлической части обода. Не на резине, не на полу.**

---

## 6. What ML does

Per inference call (один кадр):

1. Детектит колёса в кадре (single class `wheel`, multi-wheel поддерживается).
2. Эмитит per-wheel:
   - `bbox_xyxy` в пикселях;
   - `confidence` в `[0, 1]` на wheel-уровне;
   - 3 screen-space keypoint'а в порядке `a`, `b`, `c_disc_bottom`.
3. Echoes `frame_id` обратно.
4. Дропает wheel'и с:
   - partial occlusion (любой из трёх keypoint'ов ниже full-visibility-threshold);
   - геометрией A/B/C, нарушающей floor-ray инвариант (A левее B; A/B в нижней 20% bbox; C визуально выше A/B; etc. — см. `confirmed_geometry_issues()` в `src/postprocess_wheels.py`).

ML stateless per-frame: никакого внутреннего состояния между вызовами.

---

## 7. What ML does NOT do

Эти зоны принадлежат AR-стороне и **не должны ожидаться от ML**:

- Поиск пола / постановка floor plane.
- Сохранение трансформации камеры в момент захвата кадра.
- Raycast `a` и `b` в пол.
- Восстановление вертикальной плоскости колеса.
- K-frame accumulation.
- RANSAC и фильтрация outliers.
- Raycast `c_disc_bottom` на восстановленную плоскость.
- Усреднение высоты диска по valid-кадрам.
- Визуализация куба / финальных 3D-объектов с PNG-текстурами.
- Tracking / cross-frame association.
- Camera intrinsics / extrinsics / IMU.
- Main-состояние (UI, выбор колеса, цвет, «Найти ещё»).

**Forbidden fields в JSON** (catched runtime guard + test suite):

`timestamp`, `track_id`, `world_*`, `plane*`, `*_3d`, `depth*`, `*ransac*`, `*raycast*`, `intrinsic*`, `extrinsic*`, `imu*`, `visibility`, `keypoints_confidence`, `point_confidence`, `kp_confidence`.

---

## 8. How AR should consume the output

Шаги per-frame (на AR-стороне):

1. **На capture:** AR сохраняет камеру transform под ключом `frame_id` (любая стабильная строка; AR — source of truth для frame_id).
2. AR передаёт кадр + `frame_id` в ML inference.
3. ML возвращает JSON по schema из §3.
4. AR матчит ответ → camera transform по `frame_id`.
5. Для каждого `wheel` в `wheels[]`:
   1. Raycast `points.a` в найденную floor-плоскость → 3D-точка `A_world` (левый anchor).
   2. Raycast `points.b` в пол → `B_world` (правый anchor).
   3. Сохранить пару `(A_world, B_world)` в accumulator колеса (matching между кадрами по 3D-position — AR responsibility).
6. После K кадров (K — AR-side параметр):
   1. RANSAC по накопленным парам `(A, B)` для каждого колеса → одна стабильная вертикальная плоскость через RANSAC inliers, перпендикулярная полу.
   2. Для каждого inlier-кадра: raycast `points.c_disc_bottom` на восстановленную плоскость → 3D-точка диска.
   3. Усреднить 3D-точки C по inlier-кадрам → итоговая installation height.
7. На «Стоп» — заменить временный куб финальным 3D-объектом (PNG-текстура).

Особые случаи:

- **Пустой `wheels[]`** — нормальная ситуация (колесо вне кадра / неуверенная детекция / отброшенные occluded). AR должен пропустить кадр без накопления.
- **Меньше колёс, чем ожидалось** — нормально. ML агрессивно дропает occluded и геометрически невалидные. AR-side ассоциация по 3D подхватит колесо в следующем кадре.

---

## 9. How annotators should label A/B/C

Source: `docs/ANNOTATION_GUIDELINES.md` (load-bearing для annotator pass).

Краткая выжимка:

| Keypoint | Where |
|---|---|
| A (internal `rim_left`, AR-facing `a`) | В области видимого пола / земли возле левой стороны footprint колеса. Не на ободе, не на шине, не на колесе. Это пиксель, который AR raycast'ит в пол. |
| B (internal `rim_right`, AR-facing `b`) | Симметрично A — на полу возле правой стороны footprint. |
| C (internal `disc_bottom`, AR-facing `c_disc_bottom`) | Нижняя видимая граница металлического обода / диска — там, где обод встречается с резиной снизу колеса. |

Geometry consistency (annotator self-check):

- `rim_left.x < rim_right.x` (A левее B).
- `disc_bottom.y < min(rim_left.y, rim_right.y)` (C визуально выше A/B; y растёт вниз).

Drop rules:

- Колесо с partially occluded A/B/C — **drop entirely** (не угадывать).
- Spare wheel, motorbike, mural wheel, view-through-window — не размечать.
- Колесо с < 50% видимого диска — не размечать.

Internal label strings `rim_left` / `rim_right` сохранены для совместимости с converter'ом, но семантически означают **floor-ray A/B**, а не точки на ободе. Не путать.

---

## 10. How to run inference

### Single image (Ultralytics YOLO-pose)

```bash
./.venv/bin/python src/infer_image.py \
    --image data/samples/car.jpg \
    --model runs/pose/wheel_baseline/weights/best.pt \
    --conf 0.25 --iou 0.45 --max-det 20 \
    --frame-id frame_001 \
    --require-frame-id \
    --device cpu
```

Output: `outputs/car.json` — primary AR confirmed-schema JSON. Опциональные debug-артефакты: `car_legacy.json`, `car_raw.json`, `car_final_pred.jpg`.

### Batch (директория изображений или видеофайл)

```bash
./.venv/bin/python src/infer_batch.py \
    --source data/wheel_dataset/images/val \
    --model runs/pose/wheel_baseline/weights/best.pt \
    --out-dir /tmp/batch_out \
    --device cpu
```

Output: `<stem>__frame_NNNNNN.json` per кадр + `batch_summary.json` с `frame_index` manifest.

### TFLite / LiteRT runtime (Android target)

```bash
./.venv/bin/python scripts/predict_mobilenetv2_tflite.py \
    --tflite-model artifacts/mobilenetv2_skipless.tflite \
    --source data/samples/ \
    --runtime-python .tflite-venv/bin/python \
    --imgsz 640 --conf 0.30 --nms-iou 0.5 --max-det 5 \
    --out-dir outputs/mobilenetv2_tflite_runtime/handoff
```

Output: `<stem>.json` per кадр (confirmed schema) + `predictions.jsonl` + `run_summary.json` + `runtime_report.md` + previews.

Все три пути эмитят **identical confirmed JSON schema**.

---

## 11. How to run validation / tests

### Полная test-suite

```bash
./.venv/bin/pytest -q
```

Expected: `1031 passed, 0 failed` (stale: was `495 passed, 1 skipped`, 2026-05-30).

### Только contract tests (быстрый smoke)

```bash
./.venv/bin/pytest -q \
    tests/test_ar_contract.py \
    tests/test_confirmed_ar_schema_shape.py \
    tests/test_infer_batch_schema.py \
    tests/test_postprocess_wheels.py \
    tests/test_visualize_predictions.py
```

Expected: `105 passed`.

### Healthcheck (pytest + plugin synthetic ingestion smoke)

```bash
./scripts/healthcheck.sh
```

### Single-frame inference smoke

```bash
./.venv/bin/python src/infer_image.py --image data/samples/car.jpg --device cpu
cat outputs/car.json | python -m json.tool
```

### Schema validation на existing JSON

```bash
PYTHONPATH=src ./.venv/bin/python -c "
import json, sys
from postprocess_wheels import assert_confirmed_schema_closed
payload = json.load(open(sys.argv[1]))
assert_confirmed_schema_closed(payload, source_label=sys.argv[1], require_frame_id=True)
print('OK')
" outputs/car.json
```

---

## 12. Recent consolidation

Изменения, консолидированные в текущей версии репозитория:

**Removed:**

- `to_target_schema()` функция (deprecated draft schema converter).
- `INTERNAL_TO_TARGET_KP` mapping.
- `--target-schema` CLI флаг в `src/infer_image.py` и `src/infer_batch.py`.
- `<stem>_target.json` файловый output.
- Дубликат локального `_assert_confirmed_schema_closed` в `scripts/predict_mobilenetv2_skipless.py`.

**Added:**

- `assert_confirmed_schema_closed()` в `src/postprocess_wheels.py` — closed-set check на 3 уровнях (top, wheel, points) + substring sweep, единственный источник истины.
- `assert_confirmed_no_forbidden_fields()` — рекурсивный walk с substring matching, shared truth между runtime и тестами.
- `visibility_from_keypoint_confidence()` helper — централизованный visibility-threshold (full = 0.5, occluded = 0.15), без изменения numeric behavior.
- `--require-frame-id` strict mode в `src/infer_image.py` и `src/visualize_predictions.py`.
- Regression tests:
  - `test_deprecated_target_schema_converter_is_removed`
  - `test_target_schema_flag_is_removed_from_parser`
  - `test_runtime_forbidden_guard_uses_same_substring_sweep`
  - `test_runtime_forbidden_guard_rejects_nested_substring_leaks`
  - `test_visibility_from_keypoint_confidence_preserves_threshold_boundaries`
  - `test_determine_frame_id_strict_requires_explicit_value`
  - `tests/test_visualize_predictions.py` (новый файл, 3 теста)

**Strengthened:**

- Runtime forbidden-field guard теперь recursive + использует ровно тот же substring-список, что и тесты (`CONFIRMED_FORBIDDEN_KEY_SUBSTRINGS`). Гард применяется на каждом кадре во всех трёх inference-путях (Ultralytics single + batch + TFLite).
- `src/visualize_predictions.py` теперь **confirmed-schema-only**; legacy `wheel_bbox` / `keypoints` shape вызывает explicit `ValueError`.

**Documentation synced** с confirmed contract:

- `docs/DATASET_SPEC.md` — таблица keypoints, описания A/B обновлены на floor-ray, AR-facing names исправлены на `a` / `b` / `c_disc_bottom`.
- `docs/ANNOTATION_JSON_FORMAT.md` — то же, плюс example JSON значения сдвинуты под floor-ray геометрию.
- `docs/ANNOTATION_GUIDELINES.md` — annotator-facing document полностью пересобран; geometry inequality исправлено `disc_bottom.y >= max(...)` → `disc_bottom.y < min(...)`.
- `docs/ANNOTATION_TOOLING.md` — CVAT skeleton notes обновлены на legacy-names-with-floor-ray-semantics.
- `README.md`, `docs/SPEC_COMPLIANCE.md` — обновлены упоминания удалённых артефактов.

---

## 13. Known limitations

- **Internal training label names** `rim_left` / `rim_right` сохранены в коде (`src/postprocess_wheels.py:KEYPOINT_NAMES`, converter'ах и legacy YOLO label-файлах) для converter back-compat. Семантически они означают floor-ray A/B. Это **drifted naming**, документировано в `docs/KEYPOINT_SPEC.md`. Output JSON всегда использует `a` / `b` / `c_disc_bottom` — контракт не нарушен.
- **Inference latency / FPS budget** не специфицирован контрактом и не измерен в репо. Зависит от модели + железа. AR-сторона должна измерить на target-устройстве (Android first).
- **TFLite runtime запускается через subprocess** (`--runtime-python` указывает на отдельный `.tflite-venv`), чтобы не тянуть TensorFlow в основной venv. На production-Android этот subprocess pattern не применяется — там LiteRT инлайн в приложении.
- **`extract_keypoints` дублирован** между `src/infer_image.py` и `src/infer_batch.py`. Visibility-threshold helper вынесен в общий, но саму функцию решили не объединять, чтобы не сцеплять модули. Поведение идентично; pinned тестом visibility-boundary.
- **`extract_keypoints` синтезирует visibility-флаг** из YOLO per-kp confidence по правилу `vis = 2 if c >= 0.5 else (1 if c >= 0.15 else 0)`. Threshold захардкожен в shared helper. Это означает: wheel с YOLO-kp-conf < 0.5 на одном из трёх kp → дропается из confirmed output. Поведение конформно «drop on uncertainty», но не настраивается per-deployment.

---

## 14. Remaining risks

### Schema / runtime risks

1. **`"imu"` 3-letter substring в forbidden list** — latent false-positive trigger на словах `minimum`, `maximum`, `stimulus`. Сегодня неактивен (нет таких полей в payload), но при добавлении legitimate-поля с `minimum_*` именем runtime guard его отвергнет. Severity: minor.
2. **`extract_keypoints` дублирование** — drift risk низкий (поведение pinned тестами), но not ideal.
3. **TFLite-предиктор имеет mock-based test coverage** для `predict_image`; full `run()` orchestration end-to-end against actual `.tflite` weights не покрыт pytest'ом — purposeful (Tensorflow вне основного venv).

### Data / quality risks

4. **Provisional model:** текущий чекпойнт обучен на synthetic generators + provisional Unreal export (см. `docs/MOBILENETV2_LITERT_HANDOFF.md`). Geometry-correct на training-distribution; real-world generalisation **не валидирована**.
5. **No human-verified labelled batch:** `data/incoming/real_v1/` содержит 221 auto-draft через `src/auto_annotate_wheels.py`. Human QA pass через обновлённый `docs/ANNOTATION_GUIDELINES.md` обязателен перед признанием production training data.
6. **Annotation pass должен использовать обновлённый GUIDELINES** (post-2026-05-21). Бандлы, размеченные по старым (rim-edge) guidelines, **должны быть переразмечены** перед использованием.

### Integration risks

7. **AR-side 3D drift не измерен.** ML возвращает корректные screen-space точки в пределах training-distribution; насколько 3D-восстановление AR-стороны устойчиво на реальной сессии — открытый вопрос, требует measurement на AR-side после интеграции.
8. **K-frame parameter (количество кадров для RANSAC)** — AR-side выбор. Не специфицирован в ТЗ. Зависит от AR-side tracking-noise; нужен тюнинг.
9. **`frame_id` fallback:** если AR не передаёт `--frame-id` и не использует `--require-frame-id`, ML фолбэчит на image stem. В production AR должен либо всегда передавать frame_id, либо включать `--require-frame-id` (recommended).

---

## 15. Model quality status

**Status: PROVISIONAL.**

См. `docs/MOBILENETV2_LITERT_HANDOFF.md` для детального статуса.

Что есть:

- `runs/pose/wheel_baseline*/weights/best.pt` — YOLO-pose baseline, trained on synthetic + accepted Unreal export.
- MobileNetV2-skipless кастомная модель + LiteRT export — provisional handoff для AR-mobile integration smoke.
- ONNX export verified — parity drift < 2 px keypoints / < 0.05 confidence vs PyTorch.

Чего не хватает для production model approval:

1. **Human-verified annotated batch** на ≥ 200 real-world кадрах. Сейчас `data/incoming/real_v1/` — auto-drafts без QA. Human-аннотатор должен пройти через `src/manual_keypoint_annotator.py --prefill-from data/incoming/real_v1/annotations` с обновлённым `docs/ANNOTATION_GUIDELINES.md`.
2. **Retrain** на этом батче.
3. **Eval** на hold-out с метриками:
   - per-keypoint pixel error (отдельно A, B, C);
   - wheel-detection mAP;
   - geometry-filter drop rate.
4. **AR-side 3D drift measurement** после интеграции: насколько disc-bottom 3D-позиция (после raycast + RANSAC + averaging) отличается от ground truth.

Только после пунктов 1–4 модель может считаться production-ready. До этого момента AR-side тесты должны помечать predictions как **provisional**, а интеграция оставаться в режиме technical integration testing.

---

## 16. What AR team must NOT misinterpret

| Pitfall | Reality |
|---|---|
| «`points.a` — это левая точка обода» | A — пиксель в зоне видимого пола слева от wheel footprint. Старая rim-edge семантика отменена 2026-05-14. |
| «Раз `confidence` есть, можно ожидать per-keypoint confidence» | Только wheel-level. Per-keypoint confidence не эмитится. |
| «Можно использовать `<stem>_legacy.json` или `<stem>_raw.json` для production» | Debug only. Primary AR consumer = `<stem>.json` (или `<stem>__frame_NNNNNN.json` для batch). Legacy / raw содержат forbidden поля и устаревшие имена. |
| «`frame_id` ML генерирует — можно игнорировать» | AR должен передавать `frame_id` и matching transform по нему. ML только echoes back. Fallback на image stem — debug only; в production AR должен либо всегда передавать `frame_id`, либо включать `--require-frame-id`. |
| «Если пустой `wheels[]` — что-то сломалось» | Валидный response = «не обнаружено в этом кадре». AR пропускает кадр без накопления. |
| «Меньше колёс, чем в предыдущем кадре = bug» | ML агрессивно дропает occluded и геометрически невалидные wheels. Cross-frame ассоциация — AR responsibility. |
| «`bbox_xyxy` — top-left + width + height» | `bbox_xyxy` = `[x1, y1, x2, y2]` (top-left + bottom-right). Старый `bbox_xywh` draft удалён. |
| «ML может присылать `timestamp` для синхронизации» | `timestamp` исключён из contract. Синхронизация — через `frame_id`. |
| «ML может выдать `track_id`, чтобы не делать tracking на AR-стороне» | Tracking — AR responsibility, ассоциация по 3D-position после raycast. |
| «Можно попросить ML вернуть intrinsics / extrinsics / depth» | ML не имеет камеры metadata. AR owns. |

---

## 17. What annotators must NOT misinterpret

| Pitfall | Reality |
|---|---|
| «`rim_left` = left edge of metal rim» | Это legacy internal label для floor-ray A. Place в зоне видимого пола, не на ободе. |
| «A/B и C образуют треугольник на колесе» | A и B в зоне видимого пола под колесом, C на ободе. Линия A–B лежит вдоль пола; C значительно выше в screen-space. |
| «Если C не видна, можно guess» | Drop the wheel entirely. Никаких `visibility = 0 / 1` для partial occlusion в production data. |
| «Spare wheel на двери — это тоже wheel» | Spare на двери / крыше — skip. |
| «Старые bundles, размеченные по rim-edge, можно дообучить» | Re-annotate по обновлённому `docs/ANNOTATION_GUIDELINES.md`. |
| «Достаточно поставить A и B на левый / правый край шины снизу» | A/B должны быть в зоне видимого пола, чтобы raycast этих пикселей в floor-plane дал точки на полу около footprint. Шина — не пол. |
| «Geometry consistency: C должен быть ниже A/B, потому что это `disc_bottom`» | В screen-space y растёт вниз; floor-ray A/B находятся ниже диска. Поэтому `c.y < min(a.y, b.y)`. C визуально выше A/B на экране. |

---

## 18. Final readiness

| Aspect | Status | Blocking issues | Owner of next step |
|---|---|---|---|
| Contract readiness | READY | None | — |
| Code readiness | READY | None (1031 tests pass; stale: was 495) | — |
| Documentation readiness | READY | None (annotator + AR + ML docs synced) | — |
| Annotator readiness | READY | Updated `docs/ANNOTATION_GUIDELINES.md` is single source of truth | Annotators (next pass on `data/incoming/real_v1/`) |
| Model quality readiness | PROVISIONAL | (a) no human-verified batch yet; (b) no eval on real data; (c) per-kp pixel error not measured | ML team + annotators |
| Production readiness | NOT READY until real-data eval + AR-side drift measurement | All four items in §15 + AR-side 3D drift measurement (§14 risk 7) | ML team + AR team |

---

## Next steps

### ML team

1. Дождаться human-verified annotation pass на `data/incoming/real_v1/` через обновлённый `docs/ANNOTATION_GUIDELINES.md`.
2. Retrain MobileNetV2-skipless на новых данных (`scripts/train_mobilenetv2_skipless.py` + `scripts/train_compare_mobilenetv2_baseline.py`).
3. Eval по `scripts/eval_mobilenetv2_skipless.py` с pixel-error metrics по A/B/C отдельно.
4. Update `docs/MOBILENETV2_LITERT_HANDOFF.md` с post-retrain metrics.
5. Re-export TFLite/LiteRT артефакт; verify parity через `scripts/export_mobilenetv2_tflite.py` + `tests/test_export_mobilenetv2_tflite.py`.

### AR team

1. Интегрировать confirmed JSON contract (см. §8 для последовательности шагов).
2. Включить `--require-frame-id` strict mode в production inference path (рекомендовано).
3. Измерить AR-side 3D drift между recovered disc-bottom 3D-position и ground truth на сессиях, для которых есть reference.
4. Сообщить ML team K-frame parameter (для тюнинга RANSAC и при необходимости pre-filtering на ML-стороне).
5. Если K-frame требует ML-стороне фильтровать менее агрессивно (например, эмитить wheels с C visibility = 1) — это **schema change**, требует AR-team sign-off в `docs/OPEN_QUESTIONS_AR_SPEC.md` и обновления `tests/test_ar_contract.py`. Не делать unilaterally.

### Joint (ML + AR)

1. Договориться о latency budget (per-frame inference time на target Android).
2. Договориться о error budget (acceptable pixel error для A/B/C; acceptable 3D drift для disc-bottom).
3. Фиксировать дату production-go в `docs/OPEN_QUESTIONS_AR_SPEC.md` §9 после успешного eval + AR drift measurement.

---

## References

| Document | Purpose |
|---|---|
| `docs/AR_ML_CONTRACT.md` | Canonical contract. Single source of truth для JSON schema и responsibility split. |
| `docs/KEYPOINT_SPEC.md` | A/B/C геометрические определения. |
| `docs/OPEN_QUESTIONS_AR_SPEC.md` | Resolved + still-open items. Любое изменение контракта проходит здесь. |
| `docs/ANNOTATION_GUIDELINES.md` | Annotator-facing source of truth. |
| `docs/ANNOTATION_JSON_FORMAT.md` | Incoming annotation format для converter. |
| `docs/DATASET_SPEC.md` | YOLO-pose label format on disk. |
| `docs/ANNOTATION_TOOLING.md` | CVAT setup для разметчиков. |
| `docs/MOBILENETV2_LITERT_HANDOFF.md` | Текущий model status (provisional). |
| `docs/PLUGIN_DATA_EXPECTATION.md` | Что ML ждёт от collection-плагина. |
| `tests/test_ar_contract.py` + `tests/test_confirmed_ar_schema_shape.py` + `tests/test_infer_batch_schema.py` | Pinned contract; CI gate. |
| `src/postprocess_wheels.py` | Sole authoritative converter to confirmed schema + runtime guards. |
| `src/infer_image.py`, `src/infer_batch.py`, `scripts/predict_mobilenetv2_tflite.py` | Three inference entrypoints. Все идут через `postprocess_wheels`. |

---

**End of hand-off report.**

Контакт со стороны ML: VSBL ML team (через repository issues).

Любое расхождение между этим документом и `docs/AR_ML_CONTRACT.md` — последний имеет приоритет. Этот документ — snapshot на 2026-05-21.

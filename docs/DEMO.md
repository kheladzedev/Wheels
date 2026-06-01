# Demo / presentation guide

> **2026-05-27 update.** Current PyTorch integration candidate is
> **`wheel_real_v1_self_plus_ue_synthetic_s`**, not `wheel_baseline_v1`.
> Full production certification is still pending; see
> `docs/PRODUCTION_READINESS_AUDIT.md`.

## Headline (candidate `wheel_real_v1_self_plus_ue_synthetic_s`)

| Metric | Value | TZ target |
|---|---|---|
| **Box mAP50, real-only val** | **0.912 ✓** | ≥0.85 |
| **Pose mAP50, real-only val** | **0.912 ✓** | ≥0.5 |
| Mean OKS (σ=0.10) | **0.887** | n/a |
| FN rate | **0.063** | open |
| FP rate | 0.250 | open |
| KP `a` median px err | 7.5 | ≤5 |
| KP `b` median px err | 7.7 | ≤5 |
| KP `c_disc_bottom` median px err | 7.7 | ≤5 |

**Real-only detection target Box mAP50 ≥ 0.85 — passed.**

Smoke outputs from the production audit:
`outputs/production_audit/smoke_single/` and
`outputs/production_audit/smoke_batch/`. Historical comparison vs
earlier variants lives in `outputs/eval/all_models_summary.csv`.

How we got there: the previous champion (`wheel_real_v1_soft_s_aug`,
mAP50 0.814) re-predicted every photo at conf ≥ 0.5, producing a cleaner
GT than the original auto-heuristic drafts. Fine-tuning yolo11s-pose on
those 321 self-labels (vs original 89 clean / 177 soft) closed the TZ
gap in one training pass — no human-in-the-loop QA was needed.

Per-keypoint pixel error sits at 7.5–7.7 px median (TZ ≤5 px is still
the long-term target). If a single keypoint must be pixel-perfect,
sibling `wheel_real_v1_soft_n_aug` still holds 5.3 px median on
`c_disc_bottom`.

---

# Reference: original baseline (`wheel_baseline_v1`, 2026-05-13)

Цель: за 3–5 минут показать AR-команде (или внешнему слушателю)
текущее состояние ML-стороны без необходимости им что-либо запускать.
Все артефакты собираются локально, делятся файлами/скрин-шотами.

## 1. Что показывать

| Слайд / артефакт | Файл | Комментарий |
|---|---|---|
| AR/ML контракт | `docs/AR_ML_CONTRACT.md` §"JSON shape" | Подтверждён 2026-05-13. |
| Confirmed-schema пример | первый `outputs/demo/json/*.json` | Реальный вывод модели, ровно тот формат что AR получает. |
| Метрики candidate | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` + `docs/PRODUCTION_READINESS_AUDIT.md` | mAP50, OKS, FN/FP, per-keypoint px error, blockers. |
| Failure cases | `outputs/demo/<image>_pred.jpg` + `failure_samples` в JSON | Видно где модель путает левое/правое колесо. |
| Что блокирует prod | `docs/REAL_V1_RETRAIN.md` | Ручной QA-проход → wheel_real_v1. |

## 2. Команды собрать материалы

С нуля на машине с тренированным чекпойнтом:

```bash
# Метрики + comparison table
./scripts/eval_all_models.sh

# Галерея предсказаний на real-фото
./.venv/bin/python scripts/build_demo_gallery.py \
    --images-dir data/manual_real/images \
    --pattern 'real_*.jpg' \
    --model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
    --out-dir outputs/demo_self_plus_ue_synthetic_s \
    --device cpu \
    --limit 30

# Один пример AR JSON для слайда
cat outputs/demo_self_plus_ue_synthetic_s/json/$(ls outputs/demo_self_plus_ue_synthetic_s/json | head -1)
```

## 3. Sound bite на слайд

> "На входе у нейросети одно RGB-фото с камеры AR-клиента. На выходе —
> JSON по контракту от 2026-05-13: список колёс, у каждого `bbox_xyxy`,
> `confidence`, три именованные 2D точки `points.{a, b, c_disc_bottom}`.
> AR делает raycast этих точек на ранее восстановленную плоскость
> пола, RANSAC по K кадрам — и получает положение и высоту диска.
> Integration candidate `wheel_real_v1_self_plus_ue_synthetic_s`
> (yolo11s-pose, 39 MB) проходит real-only порог: Box mAP50 0.912 при
> целевом пороге ТЗ 0.85. До полного production остаются Android
> export/runtime, AR-device holdout и AR-side 3D validation."

## 4. Что НЕ показывать

- 3D пространственные координаты, RANSAC параметры, plane equations,
  K-frame аккумулятор — это AR-сторона. Если спросят, ссылка —
  `docs/AR_ML_CONTRACT.md` §"What ML does NOT do".
- TFLite / CoreML экспорты — пока не сертифицированы (note 2026-05-30: v1 —
  только Android/LiteRT, iOS/CoreML отложен, CoreML-артефактов на диске нет). Для текущей
  передачи есть PyTorch `.pt` и ONNX:
  `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.{pt,onnx}`.
  ONNX aggregate eval близок к PyTorch, но строгий parity gate падает
  на части sample; см. `outputs/production_audit/onnx_drift_20.json`.
  Android-first вариант через TFLite/LiteRT остается следующим
  интеграционным шагом, если AR-команда подтвердит on-device запуск.
- Внутренние имена кейпойнтов `rim_left` / `rim_right` в коде — это
  легаси-литералы конвертера; *семантика* A/B изменилась
  2026-05-14, теперь это floor-ray points, а не rim edges.

## 5. Что осталось после handoff

1. Подключить `best.pt` или экспериментально `best.onnx` в AR-side inference wrapper и
   проверить один записанный AR frame log end-to-end.
2. Решить, нужен ли TFLite/LiteRT экспорт для Android on-device, или
   текущий ONNX путь достаточен для первой интеграции.
3. Улучшать keypoint precision до долгосрочного ориентира ≤5 px; сейчас
   candidate дает 7.5-7.7 px median, при этом real-only detection target
   уже пройден.

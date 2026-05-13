# Demo / presentation guide — `wheel_baseline_v1`

Цель: за 3–5 минут показать AR-команде (или внешнему слушателю)
текущее состояние ML-стороны без необходимости им что-либо запускать.
Все артефакты собираются локально, делятся файлами/скрин-шотами.

## 1. Что показывать

| Слайд / артефакт | Файл | Комментарий |
|---|---|---|
| AR/ML контракт | `docs/AR_ML_CONTRACT.md` §"JSON shape" | Подтверждён 2026-05-13. |
| Confirmed-schema пример | первый `outputs/demo/json/*.json` | Реальный вывод модели, ровно тот формат что AR получает. |
| Метрики бейслайна | `outputs/eval/wheel_baseline_v1_summary.md` | mAP50, OKS, FN/FP, per-keypoint px error. |
| Failure cases | `outputs/demo/<image>_pred.jpg` + `failure_samples` в JSON | Видно где модель путает левое/правое колесо. |
| Что блокирует prod | `docs/REAL_V1_RETRAIN.md` | Ручной QA-проход → wheel_real_v1. |

## 2. Команды собрать материалы

С нуля на машине с тренированным чекпойнтом:

```bash
# Метрики + failure catalogue (≈30 сек на M3 Ultra CPU)
./scripts/eval_baseline.sh

# Галерея из 25 предсказаний на real-фото (≈3 мин)
./.venv/bin/python scripts/build_demo_gallery.py \
    --images-dir data/manual_real/images \
    --pattern 'real_*.jpg' \
    --model runs/pose/wheel_baseline_v1/weights/best.pt \
    --out-dir outputs/demo \
    --device cpu \
    --limit 25

# Один пример AR JSON для слайда
cat outputs/demo/json/$(ls outputs/demo/json | head -1)
```

## 3. Sound bite на слайд

> "На входе у нейросети одно RGB-фото с камеры AR-клиента. На выходе —
> JSON по контракту от 2026-05-13: список колёс, у каждого `bbox_xyxy`,
> `confidence`, три именованные 2D точки `points.{a, b, c_disc_bottom}`.
> AR делает raycast этих точек на ранее восстановленную плоскость
> пола, RANSAC по K кадрам — и получает положение и высоту диска.
> Сейчас бейслайн `wheel_baseline_v1` (yolo11n-pose, 5.6 MB) гонится
> ~100 мс на кадр на M3 Ultra CPU. mAP50 (box) — 0.59 на 80 авто-
> аннотированных val-фото; цель ТЗ — 0.85 после QA-прохода и
> переобучения."

## 4. Что НЕ показывать

- 3D пространственные координаты, RANSAC параметры, plane equations,
  K-frame аккумулятор — это AR-сторона. Если спросят, ссылка —
  `docs/AR_ML_CONTRACT.md` §"What ML does NOT do".
- TFLite / CoreML экспорты — пока не собраны, заблокированы Q10
  (`docs/QUESTIONS_FOR_TEAM.md`). ONNX работает,
  `runs/pose/wheel_baseline_v1/weights/best.onnx`, drift < 2 px на
  тестовом фото (`tests/test_onnx_drift.py` проверяет на каждом
  pytest run).
- Внутренние имена кейпойнтов `rim_left` / `rim_right` в коде — это
  легаси-литералы конвертера; *семантика* A/B изменилась
  2026-05-14, теперь это floor-ray points, а не rim edges.

## 5. Что блокирует переход в production

1. Ручной QA-проход по 396 авто-черновикам в
   `data/incoming/real_v1/annotations/` — запускается через
   `./scripts/qa_real_v1.sh`. Финиш на стороне человека.
2. После QA — переобучение по `docs/REAL_V1_RETRAIN.md` до
   `wheel_real_v1`; ожидаемый рост mAP50 до ~0.85.
3. Решение AR-команды по формату экспорта (Q10): останавливаемся на
   ONNX или нужен TFLite / CoreML? От этого зависит надо ли расширять
   `requirements.txt` (tensorflow / coremltools).

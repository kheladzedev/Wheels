# AR Floor / 3D Status

Дата: 2026-06-11

## Короткий статус

Мобильная часть модели уже передана на интеграцию: iOS CoreML и Android TFLite ужаты под лимит до 6 MB. iOS артефакт был проверен на iPhone, модель запускается быстро на Neural Engine.

По floor / 3D / RANSAC у нас реализована не финальная device-приемка, а вся ML-side инфраструктура для проверки:

- `src/eval3d_floorray.py` считает offline 3D-реконструкцию по floor-ray контракту: A/B в пол, RANSAC вертикальной плоскости колеса, C на плоскость, финальная 3D позиция.
- `src/validate_ar_replay.py` валидирует реальные AR replay JSONL логи с устройства: screen points, floor raycast hits, RANSAC labels, residuals, recovered plane, C-plane hit, final position.
- `src/eval_ar_replay_metric.py` считает quality metrics по реальному AR replay: inlier ratio, median/p95 residual, стабильность нормали плоскости, вертикальность плоскости, стабильность C hit и финальной 3D позиции.
- Production gate уже ожидает `outputs/production_audit/ar_3d_replay_eval.json`, поэтому эта проверка встроена в общий production evidence flow.

## Что это значит

Мы можем проверить полы и 3D ошибку численно, но только после того, как AR-интеграция отдаст реальный replay лог с телефона.

На текущий момент в проекте нет реального файла:

```text
data/incoming/ar_3d_replay/ar_replay.jsonl
```

Поэтому честный статус такой:

- tooling для проверки floor / 3D / RANSAC готов;
- локальные unit/integration tests покрывают валидатор, scorer, production gate wiring;
- реальная device-проверка еще не закрыта, потому что нужен лог из плагина/приложения.

## Что нужно от AR / Игоря

После того как плагин/механика будет готова, нужен `ar_replay.jsonl` с реального устройства. В каждой строке должен быть один wheel observation:

- `source_type`: `android_ar_device_replay`, `ios_ar_device_replay` или `ar_device_replay`;
- `screen_points`: `a`, `b`, `c_disc_bottom`;
- `floor_raycast_hits`: 3D hits для `a` и `b`;
- `inlier`;
- `residual`;
- `recovered_plane`: `normal`, `point`, `support`;
- `c_plane_hit`;
- `c_height_value`;
- `final_disc_bottom_position`.

Минимальный объем для первого прогона: 30 наблюдений, 1 сессия, желательно один стабильный сценарий на одном колесе. Лучше сразу 3-5 коротких сессий с разными ракурсами.

## Как мы это прогоняем

После получения файла:

```bash
./.venv/bin/python src/validate_ar_replay.py \
  --jsonl data/incoming/ar_3d_replay/ar_replay.jsonl \
  --out outputs/production_audit/ar_3d_replay_eval.json \
  --min-observations 30 \
  --min-sessions 1 \
  --min-floor-hit-rate 0.9 \
  --min-inlier-rate 0.7 \
  --max-median-residual 0.02 \
  --max-p95-residual 0.05 \
  --min-final-positions 1

./.venv/bin/python src/eval_ar_replay_metric.py \
  --jsonl data/incoming/ar_3d_replay/ar_replay.jsonl \
  --out outputs/ar_replay/ar_replay_metric.json \
  --per-frame-csv outputs/ar_replay/ar_replay_metric_per_frame.csv
```

## Что будет в отчете после прогона

После реального replay мы сможем сказать, где именно проблема:

- ML точки A/B/C нестабильны или смещены;
- floor raycast не попадает стабильно в пол;
- RANSAC дает мало inliers;
- residual высокий;
- recovered plane дрожит или неверно ориентирована;
- C projection прыгает по 3D;
- final disc-bottom position нестабильна.

## Формулировка для Игоря

Мобильные iOS/Android артефакты уже передали и первично проверили. Сейчас со своей стороны я закрыл tooling для проверки пола и 3D ошибки: валидатор AR replay, метрики по RANSAC/floor hit/C projection и привязку к production evidence gate.

Следующий шаг зависит от плагина: как только будет готова механика, нужен реальный `ar_replay.jsonl` с устройства. Я прогоню его через validator/scorer и верну отчет, где именно появляется ошибка: в ML точках, floor raycast, RANSAC, recovered plane или постпроцессе.

Важно: пока нет реального replay с телефона, нельзя честно сказать, что floor/3D ошибка закрыта в production. Сейчас закрыта инфраструктура проверки и готов путь приемки.

# Persistent worker and supplementary detector — measured development results

This report supersedes the previous per-crop worker timing notes in REPORT.md.
There are still NO human-reviewed labels for formal OCR/detection metrics.

## Worker lifecycle

Before: subprocess.run created a new interpreter and loaded OCR for every new
crop (~8–12 seconds). After: one locked persistent child receives JSON lines;
models initialize once. stdout is protocol-only, stderr is continuously drained.
Timeout/crash reaps child; current error is returned and next request restarts.
No forced restart per new crop. Cache key contains crop bytes, config identity,
worker script and interpreter path. Cache hits report zero inference time rather
than reusing misleading original inference latency.

Measured with cache DISABLED for 12 consecutive requests. All used PID 10744.

| Image | OCR text | Warm inference seconds (second pass) | Total crop OCR seconds (second pass) |
|---|---|---:|---:|
| 0402 | 65-L1 / 007.80 | 0.4372 | 0.4415 |
| 0403 | 83-FD / 9972 | 0.4489 | 0.4537 |
| 0404 | 95-T1 / 5167 | 0.4414 | 0.4458 |
| 0405 | 65-H6 / 9154 | 0.4252 | 0.4290 |
| 0406 | 64-F1 / 065.10 | 0.4290 | 0.4328 |
| 0407 | unreadable, invalid tail-light crop | 0.0293 | 0.0318 |

Cold initialization: 7.3134s. First total request including initialization:
7.8566s. Host preprocessing (crop PNG read/encode): ~0.0017–0.0025s for valid
crop images; worker decode ~0.0005–0.0009s. Cache-hit request: 0.0003913s,
not inference. Plate detection cost is excluded. No video FPS/realtime claim.
Raw measurements: persistent_benchmark.json.

Separate full detection-selection + OCR run (image already decoded): 0402
8.6491s cold, 0403 0.4758s, 0404 0.5224s, 0405 0.5053s, 0406 0.5206s,
0407 0.0589s (failed OCR, not successful speed sample). See
`../persistent_pipeline/results.json` and overlays/crops. Application process
startup/import time is outside this measurement.

## Detection evidence and changes

0405 original selected region joined taillight/top of plate and excluded bottom
digits. A separate Canny/closed convex quadrilateral proposal found its visible
plate frame at roughly (197,175,150,98). With outward padding and interior text
layout validation, current production selection includes both rows. Paddle now
returns `65-H6 / 9154`; previous output was unreadable. 0402/0403/0404/0406 Paddle
strings remain unchanged. Original gradient/contour detector remains available.

0407 has no closed quadrilateral detected by this supplementary branch. It
still produces a tail-light crop; Paddle returns empty. A morphological border
closing experiment also produced an almost-full-image contour on 0406 and was
NOT applied. No random threshold changes were used to force a 0407 success.

YOLO adapter is local checkpoint/plate-only and lazy; no suitable checkpoint or
human box dataset exists here. Label UI, grouped export, explicit training and
proposal precision/recall/IoU evaluator are implemented but YOLO inference and
training remain BLOCKED pending data/weights/dependency. Multi-proposal OCR API
preserves unverified status; UI still displays its single best result.

## Tests

60 pytest tests pass. Persistent fake-child tests cover same-PID reuse, cache,
config invalidation, exit/restart and timeout/recovery. Real worker run confirms
same PID across 12 requests. verify_paddle.py confirms one-line synthetic OCR
and server-side JPG upload selecting Paddle. AppTest covers existing tabs and
all OCR choices; detection label UI initial render passes. Browser file picker
and real multi-vehicle labeled accuracy are not tested. No original tracked
saved_models artifacts modified.
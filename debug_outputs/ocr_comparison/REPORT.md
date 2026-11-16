# OCR backend comparison — predictions, not measured accuracy

Environment: main Python 3.11 / TensorFlow CPU 2.16.2 / Keras 3.15.0;
isolated worker PaddleOCR 3.3.2 / PaddlePaddle 3.2.2 / OpenCV-contrib 4.10.0.84.
No original model artifacts or detector code were changed in this pass.

## Same crop pixels per backend

Each row in results.csv/json records crop SHA-256, box, timing and error/status.
The OCR timing excludes the existing plate detector. Paddle timing includes
worker startup and loading cached model files (~8–12s). Legacy model loading
is outside timing. These are cold-worker application latencies, NOT steady
state Paddle inference benchmarks. Content cache hits can be much faster.

| Image | CNN | SVM | RF | Paddle raw (row break shown as /) |
|---|---|---|---|---|
| 0402 | IILIIIIII | 11LTIII1I | 11I0III1I | 65-L1 / 007.80 |
| 0403 | IIIIIIIII | 1JFI1JIII | 111I11III | 83-FD / 9972 |
| 0404 | IIIIIIII | JJI1311I | J1I11117 | 95-T1 / 5167 |
| 0405 | ITCIIT | JEEIIT | AEEIIT | unreadable — crop incomplete, exclude |
| 0406 | IIIIIIII | 11F1II1I | 1II1I11I | 64-F1 / 065.10 |
| 0407 | IIIW1I | 1LTLTL | II5L1L | unreadable — tail-light crop, exclude |

Visual review confirms that 0402/0403/0404/0406 resemble the displayed text;
this is NOT an independent labeled test or a basis for a percentage accuracy
claim. 0405/0407 are known detector/crop failures and must not be used to rank
OCR accuracy. Detector unchanged at user request.

## Legacy audit

- CNN: input (N,32,32,1), output 36 classes; internal Rescaling(1/255), input
  raw uint8/float32 0..255, white glyphs on black. No double normalization.
- SVM/RF: class labels 0..35 map to 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.
- HOG/scaler: 324 features, SVM uses training scaler, RF uses raw HOG.
- CNN predicted I in 40 of 46 segmented components on these six candidate
  crops; RF predicted I/1 in 36 of 46. Histograms include INVALID crops and
  measure prediction collapse only, not character accuracy.
- Real glyphs are narrow/damaged and out of distribution from synthetic
  Hershey-font training. Matching polarity/shape/class order does not imply
  classifier generalization. No retraining was attempted.
- Fixed SVM score reporting: use predict_proba column of the selected class,
  not maximum probability which may refer to a different class.

## New backend

Paddle text detection is inside the existing crop; it is not a new plate
detector. Read order groups text boxes by vertical centers and sorts each
row horizontally. Raw punctuation/newlines are preserved. Text output removes
separators without ambiguous-glyph substitutions. Score is minimum token
recognition score (uncalibrated), not a heuristic or plate correctness probability.
Mobile model names are explicit; no document-orientation/unwarping weights
are loaded. Severe skew/noise remains unvalidated.

## Verification

- 52 pytest tests pass, including server-side UploadedFile render tests.
- verify_paddle.py executed real one-line OCR (51A12345) and AppTest JPEG
  upload selecting Paddle (65L100780); both passed.
- HTTP Streamlit health 200. No browser automation/file-picker test performed.
- Both environments pip check clean. No original saved_models tracked files
  modified. Character collection exports unlabeled candidates only; training
  path refuses existing/original output directories. Real-data training not run.
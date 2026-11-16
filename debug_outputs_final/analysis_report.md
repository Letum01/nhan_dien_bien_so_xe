# ALPR Debug Report (unlabeled-image diagnostics)

## Fixed test set

Images: `0402.jpg` through `0407.jpg` in `dataset/license_plate_dataset/0/images/`.
No labels were present beside the dataset images. The two visible user-provided crops support manual estimates only for 0402 (`65-L1 / 007.80`) and 0403 (`83-FD / 9972`); these are not promoted to verified benchmark labels.

## Baseline evidence

The prior saved debug output (`debug_outputs/summary.csv`) showed:

| Image | Selected baseline box | Segments | CNN raw/post |
|---|---:|---:|---|
| 0402 | 478,0,122,48 | 9 | IIIIHIIJW |
| 0403 | 127,181,83,29 | 10 | ILLTIMPIII |
| 0404 | 519,8,81,28 | 10 | IIIIFPIIJI |
| 0405 | 178,153,156,47 | 6 | ITTIIT |
| 0406 | 202,200,151,111 | 9 | IIIIIIIII |
| 0407 | 390,86,37,13 | 9 | ILIIIIIII |

Visual inspection of the proposal overlays/crops confirmed small false positives on taillights, body panels and vehicle details. On 0402/0403 the two plate rows were not consistently handled as one crop. Very narrow/tiny detections yielded plausible-looking contour counts and were mistakenly treated as plates.

## Current output (same six images, CNN)

See `summary.csv`, per-image `report.json`, and numbered intermediate images in this folder.

| Image | Selected box | Segments | CNN raw/post | Status |
|---|---:|---:|---|---|
| 0402 | 208,192,145,101 | 9 | IILIIIIII | ocr_prediction_only |
| 0403 | 209,207,177,122 | 9 | IIIIIIIII | ocr_prediction_only |
| 0404 | 141,188,179,135 | 8 | IIIIIIII | ocr_prediction_only |
| 0405 | 178,120,156,80 | 6 | ITCIIT | ocr_prediction_only |
| 0406 | 202,200,151,111 | 8 | IIIIIIII | ocr_prediction_only |
| 0407 | none | 5 | none | low_quality_candidates_rejected |

The first two boxes contain both plate rows and have high overlap with the manually estimated boxes (roughly 0.82 IoU for the estimate coordinates used in the regression tests). This validates localization/crop geometry for these two specific examples only. OCR remains incorrect on both and the other samples; no whole-dataset accuracy claim is made.

## Root-cause evidence

1. **Localization:** baseline single best contour frequently chose small/false vehicle regions. Proposal overlays show the failures. Added minimum pixel dimensions/area, multi-proposal handling and a two-row vertical/horizontal overlap merge. The main pipeline selects only plausible-sized crops with 6–11 segmented components and otherwise returns UNKNOWN.
2. **Segmentation:** original selected crops often contained only one row. Multi-proposal crops now join row boxes; small edge fragments are filtered and vertically overlapping split strokes are merged. Tests show 0402/0403 yield 9 segmented glyphs after the change.
3. **OCR:** on the same 0402/0403 crops, CNN predicts `IILIIIIII` and `IIIIIIIII`; SVM predicts `11LTIII1I` and `1JFI1JIII`; RF predicts `11I0III1I` and `111I11III`. Inverting glyph polarity worsened CNN predictions; training and inference both use white glyphs on black with values 0–255 and CNN Rescaling(1/255). This supports a synthetic-to-real generalization failure rather than polarity mismatch.
4. **Post-processing:** raw and processed strings remain identical on these failures; post-processing is not the cause.

## Changes retained / rejected

- Kept multi-proposal crop fusion, low-quality candidate rejection, edge-fragment filtering, debug artifacts and labeled regression tests.
- Tried merging vertically separated components as one glyph; this collapsed normal rows into giant bounding boxes and reduced segmentation (e.g. 0403 from 9 glyphs to 5). Reverted that rule after regression failure.
- Original model artifacts were restored from Git after detecting a prior accidental retraining, and protected in `saved_models/retraining_backup_20261116_222242/`. Current original model hashes/content are clean relative to Git; no retraining was done in this diagnostic pass.

## Evaluation limitations

`dataset/evaluation_ground_truth_template.csv` contains manual estimates for two images and blank rows for annotation. `evaluate_labeled.py` deliberately includes only `annotation_status=human_reviewed`; current evaluation has zero eligible samples. Detection PR/recall and full OCR accuracy therefore remain **unmeasured**. Do not use the synthetic character holdout scores as real-plate accuracy.
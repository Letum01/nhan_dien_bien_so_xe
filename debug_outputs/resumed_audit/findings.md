# Resumed recognition audit

## Reproduced experiments (not an independent benchmark)

No weights were trained, overwritten or restored in this pass.
`diagnose_recognition.py` reuses exactly the two development crops discussed
with the user. Full outputs are in `../geometry_diagnostic.json`.

| Image | Variant | CNN | SVM | RF |
|---|---|---|---|---|
| 0402 | existing square padding | IILIIIIII | 11LTIII1I | 11I0III1I |
| 0402 | stretched foreground | HHLILL7HL | E5LTTT7ET | E5LHLJFHL |
| 0403 | existing square padding | IIIIIIIII | 1JFI1JIII | 111I11III |
| 0403 | stretched foreground | HBFLHH77I | EHFTH4WFT | HHFUH9WFE |

Neither variant reads the visible plates (65L100780 and 83FD9972). Stretching
was therefore NOT applied to production inference. Two development examples
are insufficient to select a new preprocessing policy.

## Correction to previous segmentation interpretation

0403 contains eight visible alphanumeric characters (`83FD9972`). Nine
segmentation components do NOT prove correct extraction. An extra component
is still present. The regression test now explicitly describes this as a
known limitation, rather than reporting successful character segmentation.
0402 also has an incomplete top-row `1` component; a matching component count
alone does not prove glyph completeness.

## Debug/production discrepancy

The experimental debug branch applies rectification and different candidate
ranking than the application. Earlier top-level reports could therefore not
be treated as exact application traces. `main.end_to_end_pipeline` now exposes
an optional trace while retaining the existing four-value return signature.
Each report records `production_pipeline` separately, with actual selected
box, character boxes, raw/post text and uncalibrated classifier scores.
`08_production_*` and `production_characters/` are the corresponding evidence.
The experimental branch remains explicitly labeled as non-production.

## Evaluation correctness fixes

- Zero eligible reviewed labels => null metrics, not zero CER/accuracy.
- UNKNOWN diagnostic text is not interpreted as plate characters.
- Undecodable images remain in the denominator as missing predictions.
- Corpus CER is total edit distance / total reference characters; per-image
  mean CER remains a separately named metric.
- Detailed per-image results are included in output JSON.

The current template has two unreviewed development transcriptions. Both are
skipped, and `evaluation.json` contains zero eligible samples and null metrics.
No detector precision/recall or recognition accuracy claim is justified yet.
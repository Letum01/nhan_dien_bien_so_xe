"""Write evidence-based OCR dataset/baseline report from executed artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT/'reports/glyph_audit_20261125T070033140243Z'


def main():
    evaluation=sorted(AUDIT.glob('baseline_evaluation_*/results.json'))[-1]
    data=json.loads(evaluation.read_text(encoding='utf-8'))
    folder=evaluation.parent
    counts=data['all_class_counts']
    lines=['# Audit 25 biển OCR và baseline ký tự thật',
           '\n## Dataset — PASS số học, không tự duyệt chất lượng ảnh',
           '25 latest Glyph reviewed,204 confirmed glyphs.18trainplates/147glyphs,7validationplates/57glyphs,test0. '
           'Parent detection revision/box/group/split/hash, glyph label/reading order/text consistency PASS for all25. '
           'Source/group/exact/pHash split checks PASS. Export newly created under dataset/. '
           'Automatic train15 milestone already ran at reports/glyph_audit_20261125T065905391927Z. '
           'This audit is a new snapshot, not overwritten output.',
           '\nRaw/normalized sheets split separately in '+str(folder.relative_to(ROOT))+'. '
           'Per-plate sheets and glyph overlays are at audit root. Screenings warn on tight edges/noise, never edit or reject reviewed data automatically.',
           '\n## Class coverage',
           '|Class|Train|Validation|Test|','|---|---:|---:|---:|']
    lines += [f"|{r['class']}|{r['train']}|{r['validation']}|{r['test']}|" for r in counts]
    lines += [f"\nTrain classes: {data['train_classes']}/36. Missing: {', '.join(data['missing_train'])}.",
              f"Train1–2samples: {', '.join(data['train_1_or_2'])}.",
              '\n## Baseline classification — segmentation bypassed',
              'Evaluate exactly the57human-reviewed validation normalized glyphs. NO fitting, augmentation or calibration on validation. '
              'They come from7plates/7groups, not57independent sources. Models hashes frozen in results.json. '
              'Repository training code is synthetic but weights contain no training manifest/hash/source IDs. '
              'Historical membership therefore UNVERIFIED: ratios below are descriptive DEVELOPMENT-set correctness, '
              'not proven independent final accuracy or real-world ALPR accuracy.',
              '\n|Model|Correct/57|Descriptive accuracy|I predictions|Abstained|',
              '|---|---:|---:|---:|---:|']
    for name,r in data['models'].items():
        lines.append(f"|{name}|{r['correct']}/57|{100*r['accuracy_on_reviewed_development_glyphs']:.2f}%|{r['prediction_histogram'].get('I',0)}|{r['unknown_count']}|")
    lines += ['\nPer-class precision/recall,36/37class confusion matrices, exact prediction rows and histograms are in '+
              str(evaluation.relative_to(ROOT))+'. Ensemble abstentions count as wrong in overall denominator. '
              'Uncalibrated confidence is not used to inflate correctness. SVM support-vector proximity probe is included, '
              'but absence of a matching support vector CANNOT prove historical train disjointness.',
              '\n## Evidence and root causes',
              '- CNN predictsI53/57 despite reviewed glyph crops. Classification failure persists without automatic segmentation. '
              'SVM predictsI/1in40/57;RF41/57. Majority shares these biases,not automatically better.',
              '- Input contract remains32x32graywhite-on-black uint8;CNN uses single internal1/255;HOG324;SVM savedscaler;RF rawHOG;mapping0–35 correct. '
              'These checks rule out trivial shape/index-universal-I faults, not all model/data problems.',
              '- Synthetic fonts/strokes/noise differ from these real narrow,slanted,blurred and tight crops. This is evidence of domain sensitivity, '
              'not a controlled proof of a single causal factor. Real reviewed crop thresholding can damage strokes too.',
              '- Full plate segmentation errors (e.g.frame fragments14.jpg) remain separate. This evaluation does NOT measure detector or segmentation.',
              '\n## Human visual follow-up, NOT automatic label changes',
              'Review validation sheet#25(3),#26(0),#35(M),#2(D) for noise/tight stroke extent. '
              'Train sheet#64–71(mostly blurred small glyphs),#33–40(strong perspective/tilt),#58(N),#141(1) merit source-overlay inspection. '
              'No specific class relabel is authorized by this screening. See visual_screening_warnings in results.json for plate/file/bbox. '
              'Foreground touching boundary is common with tight crop, NOT proof of cutoff; do not reject all warnings.',
              '\n## Experiment readiness (no retraining executed)',
              'configs/ocr_real_pilot_plan.json is a PLAN, not a training instruction or promoted model. '
              'Real-only full36class run BLOCKED:15classes absenttrain,9classes1–2samples, noOCRtest. '
              'Existing real trainer also deliberately requires all36trainclasses and reviewedtest, so it is not runnable on this corpus as-is.',
              'Synthetic+real could be a controlled TRAIN-only experiment to retain classmapping, but synthetic coverage is not real evidence for missingclasses. '
              'Need user-approved experiment/provenance, deterministic generated manifest and no validation inclusion. '
              'Choose SVM/RFhyperparameters only from frozen validation; report E/K/M (validation-onlytrainunseen) separately. '
              'CNN smallbatch16,mildrotation5deg/translation.03/contrast trainonly;newweightsdirectory;compare same57developmentglyphs, '
              'then collect independent test before final claims. No epoch increase blindly.',
              'Priority data:realtrain E/K/M to support currentvalidation; then missingclassesB/G/I/J/O/Q/R/S/V/W/X/Y where valid human-readable plates exist. '
              'Add multiple independent source groups per class,especially0/O,1/I,2/Z,5/S,8/B. Do not move val to train or duplicate samples to simulate coverage.',
              '\nTraining readiness:BLOCKED for full real36class/deployment protocol. Baseline descriptive evaluation:PASS. '
              'Exploratory synthetic+real:PROPOSED/NOT_RUN. No validation fitting,no modelpromotion,no weights/YOLO/threshold/split edits.']
    (AUDIT/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(AUDIT/'FINAL_REPORT.md')


if __name__=='__main__':main()
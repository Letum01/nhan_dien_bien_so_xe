"""Summarize measured benchmark; do not promote production algorithms."""
import argparse
import json
from pathlib import Path


def main(folder):
    folder=Path(folder);report=json.loads((folder/'benchmark.json').read_text(encoding='utf-8'))
    selected=report['selected_variant_train_only']
    lines=['# Character segmentation benchmark',
           '\n25reviewedplates/204glyphs,18train147glyphs/7val57glyphs. NoOCRtest. '
           'Protocol frozen before run: maximum-cardinality one-to-one IoU>=.5, then totalIoU; no index matching. '
           'TrainF1 selection only; classifier results calculated after policy frozen. No production change.',
           '\n|Split|Variant|TP|FP|FN|P|R|F1|Matched IoU|Count equal|All matches|Full order|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split,variants in report['summary'].items():
        for name,m in variants.items():
            lines.append(f"|{split}|{name}|{m['tp']}|{m['fp']}|{m['fn']}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['mean_matched_iou']:.4f}|{m['same_count_plates']}|{m['full_match_plates']}|{m['complete_order_correct_plates']}|")
    lines+=['\nSelected:'+selected+'. Removing border-connected components was motivated by frame contamination. '
            'It can erase glyphs touching crop boundary; no claim globally safe. height_consistency scores higher onVAL '
            'but was NOT selected fromVAL; keep as diagnostic, no posthoc policy tuning.',
            '\n## Crop GT classification',
            'CNNMix immutable,32x32white-on-black,oneinternal1/255,mapping36. Validationbaseline50/53matchedglyphs '
            'andcandidate50/53; only53/57GTglyphs matched. Missing4 are FN,not classifiercorrect. '
            'GTcrop exact strings baseline3/7,candidate4/7. These are 7developmentplates used in earlier OCRvalidation, '
            'not independent finalaccuracy. Equal componentcount alone not correctness.',
            '\n## Per-plate improvement/regression based on segmentation F1',
            '|Image|Split|Baseline TP/FP/FN|Candidate TP/FP/FN|Change|','|---|---|---|---|---|']
    for base in [r for r in report['per_plate'] if r['variant']=='baseline']:
        candidate=next(r for r in report['per_plate'] if r['plate_id']==base['plate_id'] and r['variant']==selected)
        b=base['segmentation'];c=candidate['segmentation']
        def f1(s):return 2*s['tp']/max(1,2*s['tp']+s['fp']+s['fn'])
        state='improved' if f1(c)>f1(b) else 'regressed' if f1(c)<f1(b) else 'same'
        lines.append(f"|{Path(base['image']).name}|{base['split']}|{b['tp']}/{b['fp']}/{b['fn']}|{c['tp']}/{c['fp']}/{c['fn']}|{state}|")
    lines+=['\n## YOLO→segmentation→CNNMix (experimental, no GT output repair)',
            'Topmodelscore atfixed.4 selects crop. GT used ONLY afterward for detectionIoU,coverage and glyphmatching. '
            'No groundtruth glyphs/text used to fill,sort or correct output. Exact2/7developmentvalidationplates; '
            'sourceimage alreadydecoded,total includes coldisolatedYOLO startup on cachemiss,not realtime.',
            '|Image|DetectionIoU|Glyphs fully covered|Prediction|Human text|Exact|Seconds|',
            '|---|---:|---:|---|---|---|---:|']
    for r in report['yolo_end_to_end']:
        lines.append(f"|{Path(r['image']).name}|{r.get('detection_iou',0):.3f}|{r.get('gt_glyphs_fully_inside_crop',0)}/{r.get('gt_glyph_count',0)}|{r['prediction']}|{r.get('reviewed_text','')}|{r['exact_text']}|{r['total_seconds']:.3f}|")
    lines+=['\nAQUA2_20449 remainspartialcrop (5/8glyphsfullyinside),detectionIoU~.468. '
            'Othercrops may contain all glyphs but still segmentationframe/merged/stroke problems. '
            '14.jpg hasextra1likeglyph;0408wrongglyph/fragments;13.jpg count/ordering/stroke errors. '
            'MatchedIoU1.0 onmanyglyphs likely reflects annotationworkflowusingOpenCVproposals; '
            'this annotation-policy bias limits independent evidence. No automatic GT revisions.',
            '\n## Artifact definitions',
            'benchmark.json includes per-contour reasons,GT/predboxes,one-to-onematches,split/mergeoverlap '
            'candidates,unmatchedindices,readingorder andmatchedclassification. Split/merge flags are '
            'overlapdiagnostics,not human causal labels. Allcontour/redrejected maps,binaries and greenGT/redpred '
            'overlays perplate/variant. Confusion/error rows limited to geometricallymatchedglyphs; '
            'no class assigned to unmatchednoise. Summary classifier fields are computed onlybaseline/selected; '
            'zero values on other variant rows mean NOT EVALUATED,not zerocorrect. Do not compare theirOCRscores.',
            '\n## Readiness',
            'Candidate improves train and fixedvalidation segmentationF1,not enough to promote product on25development '
            'plates. Fullorder5/7stillfails2val;YOLOendtoend2/7notready. Keep baseline default;collect newindependent '
            'reviewedglyphboxes especiallyedgecutoff/frames/smalltiltedplates. No weights/classifiertraining/threshold '
            'or reviewedlabels/split changed. Testpilot notused.']
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(folder/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)
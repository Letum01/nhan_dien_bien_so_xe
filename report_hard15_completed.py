"""Postprocess fixed benchmark artifacts only; no rerun or parameter changes."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main(folder):
    folder=Path(folder)
    result=json.loads((folder/'hard15_results.json').read_text(encoding='utf-8'))
    revisions=json.loads((folder/'dataset/ocr_revisions.json').read_text(encoding='utf-8'))
    classes={split:Counter(c['label'] for r in revisions if r['split']==split for c in r['characters']) for split in ('train','val','test')}
    class_rows=[{'class':c,**{s:classes[s][c] for s in classes}} for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    with (folder/'hard15_class_counts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    errors=[]
    for r in result['records']:
        s=r['segmentation'];tags=[]
        if r['covered_glyphs']<r['gt_glyph_count']:tags.append('crop_incomplete')
        if s['fp']:tags.append('extra_or_misaligned_components')
        if s['fn']:tags.append('missing_or_misaligned_glyphs')
        if s['full_match'] and not s['complete_order_correct']:tags.append('ordering_error_all_boxes_matched')
        elif not s['complete_order_correct']:tags.append('complete_order_not_established')
        if r['classification_errors']:tags.append('classification_errors_on_valid_matches')
        errors.append({'image':r['image'],'group':r['source_group'],'crop':r['crop'],'variant':r['variant'],
                       'coverage':f"{r['covered_glyphs']}/{r['gt_glyph_count']}",'tp':s['tp'],'fp':s['fp'],'fn':s['fn'],
                       'prediction':r['prediction'],'expected':r['reviewed_text'],'exact':r['exact_text'],
                       'unmatched_pred_indices':s['unmatched_pred'],'unmatched_gt_indices':s['unmatched_gt'],
                       'classification_errors':r['classification_errors'],'tags':tags})
    with (folder/'hard15_errors.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    summary={'reviewed_plates':len(revisions),'total_glyphs':sum(len(r['characters']) for r in revisions),
             'split_plates':dict(Counter(r['split'] for r in revisions)),
             'split_glyphs':{s:sum(classes[s].values()) for s in classes},
             'train_class_coverage':sum(classes['train'][c]>0 for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
             'missing_train':[c for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' if not classes['train'][c]],
             'train_1_or_2':[c for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' if 1<=classes['train'][c]<=2],
             'summary':result['summary'],'per_plate_errors':errors,'new_contact_glyphs':len(json.loads((folder/'new_since35_contact_index.json').read_text(encoding='utf-8'))),
             'production_promoted':False,'training':False,
             'limits':'TRAIN development, 15 plates but some share source_group; prior visual inspection/ranking. No independent final test.'}
    (folder/'hard15_final.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Mốc40 OCR và benchmark hard15 — kết quả thực thi',
           f"\nSQLite snapshot: {summary['reviewed_plates']}plates/{summary['total_glyphs']}glyphs. {summary['split_plates']}; glyphs{summary['split_glyphs']}.",
           f"Train coverage {summary['train_class_coverage']}/36; missing{summary['missing_train']};1–2samples{summary['train_1_or_2']}.",
           '\nAudit full corpus/export/parent/order/source/group/hash/pHash PASS. Hard15protocol hash/code/checkpoints verified unchanged before inference. '
           'No prior completed policy-search artifact overlap detected; queue and contact sheets were viewed before, so this is not a blind independent test. '
           'No algorithm selection, training, label edit or production promotion.',
           '\n|Scope|TP|FP|FN|Precision|Recall|F1|Exact text|Full match/order|Matched CNN|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,m in result['summary'].items():
        lines.append(f"|{key}|{m['tp']}|{m['fp']}|{m['fn']}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['exact_text']}/15|{m['full_match_order']}/15|{m['matched_classifier_correct']}/{m['matched_classifier_samples']}|")
    lines+=['\nGTcrop no padding;YOLO crop highestscore>=.4 +15%eachside fixed. All15YOLOpaddedcrops contain all reviewed glyphboxes; '
            'coverage does NOT prove segmentation or recognition correctness. MatchedCNN excludes FP/FN and must be interpreted together with segmentation recall.',
            '\n## Per-plate errors (no index pairing)',
            '|Image / group|Crop|Variant|TP/FP/FN|Prediction|Human text|Error tags|','|---|---|---|---|---|---|---|']
    for e in errors:
        lines.append(f"|{Path(e['image']).name} / {e['group']}|{e['crop']}|{e['variant']}|{e['tp']}/{e['fp']}/{e['fn']}|{e['prediction']}|{e['expected']}|{', '.join(e['tags'])}|")
    lines+=['\n## Key failures and regressions',
            '- 0409 GT:baseline9TP/0FN and exact text; border removes4GT glyphs, leaving5TP/4FN. Same image YOLO crop exact under both. Border rule is crop-boundary dependent, not generally safe.',
            '- AQUA4_null:both methods GT/YOLO essentially fail segmentation (0TP), despite full coverage. Inspect binary/connected-frame/lowcontrast evidence; no tuning here.',
            '- 20.jpg,19.jpg and2.jpg:noise/misaligned components and ordering remain; classifier correctness on matchedglyphs does not repair missing/extra output.',
            '- Confusions E→F,B→8,S→5,9→U/Q and4→I occur on valid geometry matches. No syntax corrections using GT.',
            '\n## Artifacts',
            'hard15_results.json includes one-to-one pairs,GTmissing/predextra indices,contour rejection reasons,per-group metrics and timings. '
            'hard15_errors.csv,hard15_class_counts.csv,hard15_final.json provide summaries. '
            'hard15/plateID/ contains greenGT/redprediction overlays,crops,binaries; new_since35_contact.jpg and indexJSON contain40glyphsfrom5newplates. '
            'Full corpus raw/normalized perplatecontact sheets and dataset/ retain all327glyphs.',
            '\n## Next step',
            'Keep all production settings. Use remaining hard cases as new predeclared evaluation cohorts before more tuning. '
            'Study boundary-connected trueglyphs separately from plate frames and lowcontrast fused components; annotate noise/frame categories manually if needed. '
            'Shared source groups mean15plates are not15independent sources. No OCRtest corpus; no generalization or deployment claim.']
    (folder/'HARD15_FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:summary[k] for k in ('reviewed_plates','total_glyphs','split_glyphs','train_class_coverage','missing_train','train_1_or_2','new_contact_glyphs')})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)
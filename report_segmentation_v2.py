"""Summarize fixed development ablation, never selects from validation."""
import json
import csv
from pathlib import Path
from glyph_review import ROOT


def main():
    path=sorted((ROOT/'reports').glob('segmentation_v2_*/results.json'))[-1]
    data=json.loads(path.read_text(encoding='utf-8'));folder=path.parent
    rows=data['records'];lines=['# Segmentation V2 — TRAIN development only',
        '\n33 TRAIN plates/270 glyphs. Old7validation and pending13hard crops excluded from tuning. GTcrop vs YOLOscore.4 pad15 separate. '
        'CNN fixed; max-cardinality IoU>=.5 matching. No OCR fitting, labels changed, or production promotion.',
        '\n|Crop|Variant|TP|FP|FN|P|R|F1|Exact/33|CER|Fullorder|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind,variants in data['summary'].items():
        for variant,m in variants.items():
            lines.append(f"|{kind}|{variant}|{m['tp']}|{m['fp']}|{m['fn']}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['exact']}|{m['cer']:.4f}|{m['full_order']}|")
    lines+=['\nTrain-only ranking selected: **'+data['selected_train_only']+'**. Exact then CER, recall,F1. This is development fitting, not generalization evidence.',
            '\n## Mechanisms and limitations',
            '- A uses long directional opening restricted to12%perimeter bands; removes mask pixels, not entire border-connected CC.',
            '- B protects CC height18–75%crop and aspect.10–1.25. Narrow glyphs allowed; fused glyph+frame may fail CC protection.',
            '- C uses CLAHE+adaptive21/C7 then bounded line masking. D uses largest row-center gap >.55median glyphheight then left-right sorting.',
            '- No projection split was added without demonstrated merge-safe criterion; no combined variant was searched after results.',
            '- Mask hits on ink within reviewed boxes is only a risk proxy, not confirmed true-stroke erasure; background/frame can occur inside boxes.',
            '- FN differences identify lost geometry matches, NOT a guaranteed count of glyphs physically erased. No manual noise/frame labels.',
            '- No internal held-out development subgroup used: all33 TRAIN sources are development. Source-group summaries provided; no SQLite split change.',
            '\n## 0409 regression and low contrast evidence']
    errors=[]
    for r in rows:
        m=r['matching'];base=next(x for x in rows if x['plate_id']==r['plate_id'] and x['crop']==r['crop'] and x['variant']=='baseline')
        errors.append({'image':r['image'],'group':r['group'],'crop':r['crop'],'variant':r['variant'],
                       'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'extra_pred_indices':str(m['unmatched_pred']),
                       'missing_gt_indices':str(m['unmatched_gt']),'split_overlap_candidates':str(m['split_overlap_candidates']),
                       'merge_overlap_candidates':str(m['merge_overlap_candidates']),
                       'prediction':r['prediction'],'truth':r['truth'],'relative_order':m['relative_order_correct_on_matches'],
                       'delta_fn_vs_baseline':m['fn']-base['matching']['fn'],'seconds':r['seconds']})
        if Path(r['image']).name=='0409.jpg' or 'AQUA4_null_checkoutex' in r['image']:
            lines.append(f"- {Path(r['image']).name} {r['crop']} {r['variant']}: {m['tp']}/{m['fp']}/{m['fn']} TP/FP/FN; `{r['prediction']}` vs `{r['truth']}`; mask ink-intersection{r['mask_hits_reviewed_ink_pixels']}")
    with (folder/'error_breakdown.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    from run_segmentation_v2 import aggregate
    groups={g:{kind:{v:aggregate([r for r in rows if r['group']==g and r['crop']==kind and r['variant']==v]) for v in data['summary'][kind]} for kind in data['summary']} for g in sorted({r['group'] for r in rows})}
    (folder/'per_source_group.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
    lines+=['\n## Artifacts',str(folder),'\ncomparison.csv,results.json,error_breakdown.csv,per_source_group.json;eachplate masks/overlay greenGT redprediction.',
            '\n## Decision','Keep production unchanged. If no V2 beats border under exact-first rule, explicitly keep border as research control; hard13 gate remains BLOCKED until all13real reviews. No model retraining.']
    text='\n'.join(lines)
    (folder/'REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'reports/SEGMENTATION_V2_REPORT.md').write_text(text,encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
"""Conservative row ordering ablation; never inserts or removes a detection."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from hard_negative_data import ROOT, OUT
from segmentation_benchmark import match_boxes
from acceptance_pilot import edit_counts


def row_order(boxes):
    n=len(boxes)
    if n<2:return list(range(n))
    center=[b[1]+b[3]/2 for b in boxes]
    ids=sorted(range(n),key=center.__getitem__)
    gap=np.diff([center[i] for i in ids]); split=int(np.argmax(gap))+1
    if min(split,n-split)>=2 and max(gap)>.65*np.median([b[3] for b in boxes]):
        rows=(ids[:split],ids[split:])
    else:
        return list(range(n))  # Retain the detector pipeline's existing order when ambiguous.
    return [i for row in rows for i in sorted(row,key=lambda k:boxes[k][0])]


def main():
    source=json.loads((ROOT/'reports/accuracy_recovery/benchmark.json').read_text(encoding='utf-8'))
    records=[]
    for row in source['records']:
        if row['arm']!='baseline':continue
        start=perf_counter(); boxes=row['pred_boxes']; raw=row['ocr']['cnn']['prediction']; ids=row_order(boxes)
        value=''.join(raw[i] for i in ids); elapsed=perf_counter()-start
        records.append({'image':row['image'],'reference':row['reference'],'raw':raw,'candidate':value,'order':ids,
                        'geometry':match_boxes([boxes[i] for i in ids],row['gt_boxes']),'seconds_order_only':elapsed})
    n=sum(len(r['reference']) for r in records)
    summary={'exact':sum(r['reference']==r['candidate'] for r in records),
             'cer':sum(sum(edit_counts(r['reference'],r['candidate']).values()) for r in records)/n,
             'changed':sum(r['raw']!=r['candidate'] for r in records),
             'regressions':sum(r['reference']==r['raw'] and r['reference']!=r['candidate'] for r in records),
             'full_order':sum(r['geometry']['complete_order_correct'] for r in records)}
    path=Path(OUT/'evaluation/row_probe.json')
    with path.open('x',encoding='utf-8') as f:json.dump({'summary':summary,'records':records},f,indent=2)
    print(summary)


if __name__=='__main__':main()
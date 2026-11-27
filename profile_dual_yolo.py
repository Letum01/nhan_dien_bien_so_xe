"""Compare against the saved pre-optimization Round 1 benchmark on all 12 val plates."""
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

from benchmark_yolo_character import score,summarize
from dual_yolo_client import Worker,ROOT
from dual_yolo_pipeline import run,MixClassifiers


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)))


def main():
    baseline=json.loads((ROOT/'experiments/yolo_character_round1/comparison/results.json').read_text(encoding='utf-8'))
    old={r['id']:r for r in baseline['records'] if r['scope']=='yolo' and r['method']=='character_yolo'}
    rows=[r['review'] for r in json.loads((ROOT/'datasets/yolo_character_round1/manifest.json').read_text(encoding='utf-8'))['rows'] if r['review']['split']=='val']
    assert len(rows)==12 and set(old)=={r['image_id'] for r in rows}
    t=perf_counter(); models=MixClassifiers(); ocr_load=perf_counter()-t
    images=[cv2.imread(str(ROOT/r['image_path'])) for r in rows]; output=[]; cold=[]; single=[]
    with Worker() as worker:
        for row,image in zip(rows,images):
            a=run(image,worker,models,'all'); ref=old[row['image_id']]; plate=a['plate']
            box=plate['crop_box'] if plate else None
            scored=score(a,box,ref['gt_boxes'],row)
            checks={'plate_bbox':(plate and plate['bbox'])==(ref['plate_detection'] and ref['plate_detection']['bbox']),
                    'crop':box==ref['crop_box'],'ordered_glyph_boxes':scored['pred_boxes']==ref['pred_boxes'],
                    'texts':all(v['prediction']==ref['ocr'][k]['prediction'] for k,v in scored['ocr'].items()),
                    'score_delta':max((abs(x-y) for x,y in zip(a['glyph_scores'],ref['glyph_scores'])),default=0)}
            output.append({**scored,**a,'image':row['image_path'],'checks':checks})
        for image in images:
            a=run(image,worker,models,'cnn'); single.append(a)
        for a,b in zip(single,output): assert a['text']['cnn']==b['text']['cnn'] and a['glyph_boxes']==b['glyph_boxes']
    cold.append(output[0])
    for image in images[:2]:
        with Worker() as worker: cold.append(run(image,worker,models,'all'))
    def timings(records):return {k:stats([r['timings'][k] for r in records if k in r['timings']]) for k in sorted({k for r in records for k in r['timings']})}
    result={'legacy_wall':stats([r['seconds'] for r in old.values()]),'ocr_model_load_seconds':ocr_load,
            'cold':timings(cold),'startup':{k:stats([r['worker']['startup'][k] for r in cold]) for k in cold[0]['worker']['startup']},
            'python_startup':stats([r['worker']['python_startup_seconds'] for r in cold]),'warm_all':timings(output[1:]),'warm_cnn':timings(single),
            'regression_pass':all(all(v for k,v in r['checks'].items() if k!='score_delta') for r in output),
            'summary':summarize([dict(r,seconds=r['timings']['total']) for r in output]),'records':output,'single_memory':[r['memory'] for r in single]}
    (ROOT/'reports/dual_yolo_profile.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('records','single_memory')})


if __name__=='__main__':main()
"""Draw per-backend learning curves from comparable, completed development runs."""
import argparse
import csv
from html import escape
import math
from pathlib import Path

from final_ocr_dataset import read


def collect(paths):
    series = [read(p) for p in paths]
    def signature(r):
        return r['benchmark_sha256'], r['backend'], r['features'], tuple(
            (k,r['contract'][k]) for k in ('protocol','curation','code','sampler','batch'))
    if not series or len({signature(r) for r in series})!=1:
        raise ValueError('Use the same cohort, protocol, backend, features and recipe for each curve')
    if any(r.get('test_inference') is not False for r in series):
        raise ValueError('Only development runs can be used for this curve')
    plate_ids={r.get('plate_cohort_sha256') for r in series}
    plate_metrics=bool(len(plate_ids)==1 and None not in plate_ids and '' not in plate_ids)
    fields=('glyph_count','accuracy','macro_f1','cer','exact_match','training_seconds')
    output=[]
    for r in sorted(series,key=lambda row:row['glyph_count']):
        row={k:r.get(k) for k in fields}
        if not plate_metrics: row.update(cer=None,exact_match=None)
        if any(not math.isfinite(v) or v<0 for v in row.values() if v is not None):
            raise ValueError('Invalid metric')
        if output and row['glyph_count']<=output[-1]['glyph_count']:
            raise ValueError('Two runs have the same actual size; keep one by the registered rule')
        output.append(row)
    return output, f"{series[0]['backend']} / {series[0]['features']}"


def render(rows,title,output):
    axes=('accuracy','macro_f1','cer','exact_match','training_seconds')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="1040">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="48" y="28" font-size="18">{escape(title)} — development</text>']
    max_x=max(r['glyph_count'] for r in rows)
    for i,key in enumerate(axes):
        y=68+i*196; y0=y+130
        svg.append(f'<text x="48" y="{y-9}">{key}</text>')
        points=[(r['glyph_count'],r[key]) for r in rows if r[key] is not None]
        if not points:
            svg.append(f'<text x="48" y="{y+30}">Not measured on a common cohort</text>'); continue
        max_y=max([v for _,v in points]+[1.]) if key in ('cer','training_seconds') else 1.
        pos=[(48+720*x/max_x,y0-130*v/max_y) for x,v in points]
        svg.append(f'<path d="M48 {y} V{y0} H768" fill="none" stroke="black"/>')
        svg.append('<polyline fill="none" stroke="#2765b8" points="'+ ' '.join(f'{a:.2f},{b:.2f}' for a,b in pos)+'"/>')
        for (a,b),(n,v) in zip(pos,points):
            svg.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="4" fill="#2765b8"><title>{n} glyphs: {v:.5g}</title></circle>')
        svg.append(f'<text x="48" y="{y0+18}" font-size="12">Train glyphs: 0 to {max_x}; vertical scale: 0 to {max_y:.3g}</text>')
    with output.open('x',encoding='utf-8') as f: f.write('\n'.join(svg+['</svg>']))
    with output.with_suffix('.csv').open('x',encoding='utf-8',newline='') as f:
        table=csv.DictWriter(f,fieldnames=list(rows[0])); table.writeheader();table.writerows(rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs',type=Path,nargs='+'); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.suffix.lower()!='.svg': ap.error('--output must end in .svg')
    render(*collect(a.runs),a.output)
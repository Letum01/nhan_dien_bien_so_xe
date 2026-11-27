"""Read-only integrity comparison and executed check annotations."""
import hashlib
import json
import sqlite3
from pathlib import Path
from glyph_review import ROOT

p=ROOT/'reports/segmentation_v2_20261125T120612083313Z'
integrity={}
for name in ('annotations.sqlite3','ocr_glyphs.sqlite3'):
    with sqlite3.connect(ROOT/'reports/glyph_audit_20261125T103134292702Z'/name) as old,sqlite3.connect(ROOT/'data'/name) as now:
        integrity[name]={t:old.execute('select * from '+t+' order by rowid').fetchall()==now.execute('select * from '+t+' order by rowid').fetchall() for t in ('revisions','drafts')}
protocol=json.loads((ROOT/'reports/hard13_v2_protocol.json').read_text(encoding='utf-8'))
integrity['weights']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()==s for name,s in protocol['checkpoints_sha256'].items()}
(p/'integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
text='''

## Executed checks
133 pytest PASS; 327glyph validator PASS; fresh full audit parent/leakage PASS;
Chromium YOLO upload/fourOCR/Diagnostics PASS; both pip checks PASS; diff check PASS.
Hard13 gate executed and correctly BLOCKED13missing real reviews. No new model fitting.
See integrity.json for unchanged SQLite revisions/drafts and CNN/YOLO checkpoint hashes.

## Hard13 protocol
reports/hard13_v2_protocol.json: 13pendingTRAINplates, 0 CNNmix real-training source
exposures, 13 YOLOround2 training source exposures (sourceID check). Prior synthetic
artifact provenance is unverified. Queue hardness/crops had been proposed before;
not an independent final test. NoV2 wins exact-first: border remains research control,
NOT product default. Frozen gate checks parent/hash and refuses partial evaluation.

## Evidence and decision
0409GT: A/B/C/D recover9TP0FN vs border5TP4FN, but YOLOcrop V2 has8TP1FN vs
border9TP0FN. Mask visual shows local perimeter pixels removed, not whole component.
AQUA4_null: C recovers8TP0FP0FN on bothcrops, but aggregate YOLO exact2/33
vs border18/33. Never select a candidate solely from one repaired case.
No combined/search follow-up was attempted. Keep every V2 experimental, no promotion.

Mask/ink intersections are a risk proxy, NOT confirmed true-stroke removal count;
bbox labels cannot distinguish frame pixels inside a glyphbox. FN differences are
geometry regressions, not all physical glyph deletion. Timings perrecord exclude
YOLO worker startup, and all396 segmentation/CNN loops total37.68seconds; not realtimeFPS.

## Remaining work
Manually distinguish frame/true-stroke connectivity on reviewed crops and validate
safe line junction breaking. Avoid repeated tuning on the same33sources. Projection
splitting and combined variants were not added without merge-safe evidence. Preserve
13pending sources until human glyph review, with source exposure caveats above.
'''
for report in (p/'REPORT.md',ROOT/'reports/SEGMENTATION_V2_REPORT.md'):
    with report.open('a',encoding='utf-8') as stream:stream.write(text)
print(json.dumps(integrity,indent=2))
"""Prepare human review queue, no OCR pseudo-labels or review writes."""
import json
from pathlib import Path
from annotation_store import AnnotationStore
from glyph_review import ROOT,plates_from_detection


def main():
    entries=plates_from_detection(AnnotationStore(ROOT/'data/annotations.sqlite3'))
    confirmed=[r for r in entries if r['text_confirmed']]
    remaining=[r for r in entries if not r['text_confirmed']]
    chosen=confirmed[:20]
    # Add source/layout variety as metadata proxies, not new labels.
    for row in remaining:
        if len(chosen)>=20:break
        if row['source_group'] not in {r['source_group'] for r in chosen}:chosen.append(row)
    report={'status':'QUEUE_ONLY_NOT_REVIEWED','plate_ids':[r['image_id'] for r in chosen],
            'confirmed_texts_available':len(confirmed),'entries':chosen,
            'glyph_labels_auto_generated':False}
    (ROOT/'reports/glyph_pilot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Detection plates',len(entries),'text confirmed',len(confirmed),'pilot',len(chosen))


if __name__=='__main__':main()
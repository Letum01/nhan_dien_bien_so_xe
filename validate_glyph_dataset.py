"""Validate OCR glyph export metadata, pixel contracts and split isolation."""
import argparse
from collections import Counter
import csv
from pathlib import Path

import cv2
import numpy as np


def validate(root):
    root=Path(root)
    with (root/'characters.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Empty glyph dataset')
    groups={};sources={};seen=set();counts=Counter()
    for row in rows:
        if row['annotation_status']!='human_reviewed' or len(row['class_label'])!=1 or row['class_label'] not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            raise ValueError('Unreviewed/invalid class')
        path=(root/row['image_path']).resolve()
        if root.resolve() not in path.parents or path in seen:raise ValueError('Duplicate/unsafe glyph path')
        seen.add(path)
        image=cv2.imread(str(path),0)
        if image is None or image.shape!=(32,32) or not set(np.unique(image))<={0,255}:raise ValueError('Glyph pixel contract invalid')
        for mapping,key in ((groups,row['source_group']),(sources,row['source_image_id'])):
            if not key or key in mapping and mapping[key]!=row['split']:raise ValueError('Source/group split leakage')
            mapping[key]=row['split']
        if row['split'] not in ('train','validation','test'):raise ValueError('Invalid split')
        counts[row['class_label']]+=1
    return {'status':'PASS','glyphs':len(rows),'class_counts':dict(counts),
            'missing_classes':[c for c in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' if not counts[c]],
            'training_readiness':'NOT_ASSERTED; validation PASS does not imply sufficient samples/classes'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();print(validate(args.root))
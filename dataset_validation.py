"""Strict source/group/duplicate validation and reproducible group split plans."""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import random

from dataset_audit import fingerprint


def validate_boxes(boxes, width, height):
    from localization import _box_iou
    for i, box in enumerate(boxes):
        if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
            raise ValueError("Box must contain four finite numbers")
        x,y,w,h=box
        if min(x,y)<0 or min(w,h)<=0 or x+w>width or y+h>height:
            raise ValueError("Box outside source image")
        if min(w,h)<2:
            raise ValueError("Degenerate plate box below 2 pixels; human correction required (do not auto-delete)")
        if any(_box_iou(box, other)>.95 for other in boxes[:i]):
            raise ValueError("Duplicate annotation boxes")


def check_split_leakage(rows, root):
    groups, exact, perceptual = {}, {}, []
    for row in rows:
        path=Path(root)/row["image_path"]
        digest,phash,shape=fingerprint(path)
        if shape is None:
            raise ValueError(f"Corrupt image: {path}")
        group,split=row.get("source_group"),row.get("split")
        if not group or split not in ("train","val","validation","test"):
            raise ValueError("source_group/split required")
        split="val" if split=="validation" else split
        if group in groups and groups[group]!=split:
            raise ValueError("Source group leakage")
        if digest in exact and exact[digest]!=split:
            raise ValueError("Exact duplicate leakage")
        for other,old_split in perceptual:
            if phash is not None and (phash^other).bit_count()<=6 and split!=old_split:
                raise ValueError("Near-duplicate leakage candidate: review/group before export")
        groups[group]=split;exact[digest]=split
        if phash is not None:
            perceptual.append((phash,split))


def group_split_plan(rows, seed=42):
    """Deterministic source-group split proposal (not duplicate union).

    Returns a plan only; never edits reviewed annotations or preexisting test.
    Tiny datasets may not populate all splits: validator must reject training.
    """
    groups=sorted(set(row["source_group"] for row in rows))
    if any(not group for group in groups):
        raise ValueError("Missing source group")
    random.Random(seed).shuffle(groups)
    n=len(groups);train_end=int(n*.70);val_end=train_end+int(n*.15)
    mapping={group:("train" if i<train_end else "val" if i<val_end else "test") for i,group in enumerate(groups)}
    return {"seed":seed,"ratios":[.70,.15,.15],"groups":mapping,
            "rows":[{**row,"split":mapping[row["source_group"]]} for row in rows],
            "warning":"Run check_split_leakage after grouping perceptual/exact duplicate candidates. Plan never overwrites established test labels."}
"""Pure editing operations for the unified annotation UI; no model labels."""
import copy

from dataset_validation import validate_boxes


def split_character(characters,index,offset):
    result=copy.deepcopy(characters)
    old=result.pop(index)
    x,y,w,h=old['bbox']
    if not 0<offset<w:
        raise ValueError("Split must lie strictly inside the selected glyph")
    for delta,width in reversed([(0,offset),(offset,w-offset)]):
        item={**old,'bbox':[x+delta,y,width,h],'label':'','confirmed':False}
        result.insert(index,item)
    return reorder(result)


def merge_characters(characters,indices):
    if len(set(indices))<2:
        raise ValueError("Select at least two glyphs")
    selected=[characters[i] for i in sorted(set(indices))]
    if len({item.get('line',0) for item in selected})!=1:
        raise ValueError("Cannot merge characters across different rows")
    left=min(c['bbox'][0] for c in selected);top=min(c['bbox'][1] for c in selected)
    right=max(c['bbox'][0]+c['bbox'][2] for c in selected);bottom=max(c['bbox'][1]+c['bbox'][3] for c in selected)
    merged={**selected[0],'bbox':[left,top,right-left,bottom-top],'label':'','confirmed':False}
    result=[copy.deepcopy(c) for i,c in enumerate(characters) if i not in indices]+[merged]
    return reorder(result)


def reorder(characters):
    result=sorted(copy.deepcopy(characters),key=lambda c:(int(c.get('line',0)),c['bbox'][0]))
    for i,char in enumerate(result):char['order']=i
    return result


def sync_characters(characters,boxes):
    """Match by canvas order. Geometry change invalidates human label approval."""
    result=[]
    for index,box in enumerate(boxes):
        old=characters[index] if index<len(characters) else {}
        same=old.get('bbox')==box
        result.append({**old,'bbox':box,'label':old.get('label','') if same else '',
                       'confirmed':bool(old.get('confirmed')) if same else False,
                       'line':old.get('line',0),'order':index,'invert':old.get('invert',False)})
    return result


def validate_plate_text(plate):
    if not plate.get('text_confirmed') or plate.get('unreadable'):
        raise ValueError("Plate text must be explicitly confirmed and readable")
    if not plate.get('plate_text','').strip():raise ValueError("Missing plate text")
    # Preserve original human text; this check never rewrites B/O/etc.
    if any(not(ch.isascii() and (ch.isalnum() or ch in ' .-/\n')) for ch in plate['plate_text']):
        raise ValueError("Unsupported text character; correct manually or mark unreadable")
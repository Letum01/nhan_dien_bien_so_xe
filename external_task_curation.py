"""Read-only task-specific OCR triage; never converts heuristics to supervision."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from external_dataset_prepare import (AUDIT, CLASSES, ROOT, TAGS, crop, digest, draw,
                                      iou, load_image, mapping, protected, read, safe_path, sheet,
                                      vn_box, write)
from final_ocr_dataset import check_protocol
from yolo_character_pipeline import normalize_glyph

PINNED = {
    'summary.json': '25d5ff37cc4067df0a31e1eac8bdbae7b5fc528dffe7bcfafecaf33c486923cd',
    'class_mapping.json': 'bb58385d1565b30f14d7ccbab56573e83370f656c4829abeb48e25b9f13061c3',
    'glyph_review_manifest.json': '2da6fd97de067adbfc6b977717eefca80bfc04791b8c016f1a5e053a00c5b138',
    'leakage.json': 'c7a712d056ba35e20550458ef0caff41a8a6f4b257c66ece09412c3c1f109249',
}
TASKS = ('CLASSIFIER_ELIGIBLE', 'CHARACTER_DETECTION_ELIGIBLE', 'FULL_PLATE_EVALUATION_ELIGIBLE')


def glyph_flags(im, box, others):
    x, y, w, h = box
    H, W = im.shape[:2]
    if not all(math.isfinite(v) for v in box) or min(w, h) <= 0 or min(x, y, W-x-w, H-y-h) < -.002:
        return None, ['invalid_box']
    roi, pixel = crop(im, box)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = float(np.ptp(np.percentile(gray, [5, 95])))
    norm = normalize_glyph(roi)
    fraction = float(np.count_nonzero(norm)/norm.size)
    resized = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    border = np.r_[mask[0], mask[-1], mask[1:-1,0], mask[1:-1,-1]]
    if np.mean(border > 0) > .5:
        mask = cv2.bitwise_not(mask)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    substantial = [c for c in stats[1:] if c[cv2.CC_STAT_HEIGHT] >= .5*gray.shape[0]
                   and c[cv2.CC_STAT_AREA] >= .04*gray.size]
    overlap = max((iou(box,b) for b in others),default=0)
    flags = []
    if min(pixel[2:]) < 5 or pixel[3] < 10: flags.append('small_pixel_roi')
    if contrast < 20 or not .015 <= fraction <= .8: flags.append('unreliable_foreground_proxy')
    if sharpness < 12: flags.append('blur_proxy')
    if min(x,y,W-x-w,H-y-h) <= 1: flags.append('parent_edge_risk')
    if overlap >= .2: flags.append('box_overlap')
    if len(substantial) >= 2: flags.append('multiple_tall_components')
    if np.mean(np.r_[mask[0],mask[-1]] > 0) > .45: flags.append('stroke_cut_risk')
    return {'contrast_p95_p05':contrast,'normalized_ink_fraction':fraction,'laplacian_var_32':sharpness,
            'tall_components':len(substantial),'max_iou':overlap,'pixel_xywh':pixel}, flags


def classify_tier(flags, quarantined):
    if quarantined or 'invalid_box' in flags:
        return 'D'
    return 'C' if flags else 'B'


def observe_vnlp(root, rows, output):
    tiles, notes = [], []
    for row in rows:
        if row['width'] >= row['height']:
            continue
        path = safe_path(root,row['path'])
        if digest(path) != row['sha256']:
            raise ValueError(f'VNLP portrait image differs from existing audit: {path}')
        raw = load_image(path)
        with Image.open(path) as img:
            exif = img.getexif().get(274,1)
        h,w = raw.shape[:2]
        oriented = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        xyxy = vn_box(row['filename_coords'],w,h,True)
        xywh = vn_box(row['filename_coords'],w,h,False)
        def overlay(im, xywh, xyxy):
            if xywh: draw(im,[['xywh',*xywh[1:]]],(0,180,255))
            if xyxy: draw(im,[['xyxy',*xyxy[1:]]],(0,255,0))
            return im
        key = f'{len(notes)+1:02d}'
        if len(notes)<12:
            tiles.append((f'{key} raw / EXIF {exif}',overlay(raw.copy(),xywh,xyxy)))
            oh,ow = oriented.shape[:2]
            a,b = (vn_box(row['filename_coords'],ow,oh,mode) for mode in (False,True))
            tiles.append((f'{key} EXIF decode / same coords',overlay(oriented,a,b)))
        notes.append(dict(id=key,path=row['path'],sha256=row['sha256'],raw_wh=[w,h],exif=exif,
                          decoded_wh=list(oriented.shape[:2][::-1]),xywh=xywh,xyxy=xyxy))
    write(output/'vnlp_portrait.json',notes)
    sheet(output/'vnlp_portrait.jpg',tiles)
    return {'checked_portrait':len(notes),'exif_orientation':dict(Counter(r['exif'] for r in notes)),
            'changed_decode_size':sum(r['raw_wh']!=r['decoded_wh'] for r in notes),
            'coordinate_resolution':'UNVERIFIED; no conversions exported'}


def curate(prep, ext, out):
    check_protocol()
    for name, expected in PINNED.items():
        if digest(prep/name)!=expected: raise ValueError(f'Pinned input changed: {name}')
    summary=read(prep/'summary.json'); source_map=read(prep/'class_mapping.json')
    forward,reverse = mapping({'nc':len(source_map['source_order']),'names':source_map['source_order']})
    if source_map['target_order'] != CLASSES or forward != {int(k):v for k,v in source_map['source_to_target'].items()} or reverse != {int(k):v for k,v in source_map['target_to_source'].items()}:
        raise ValueError('Invalid mapping contract')
    for tag in TAGS:
        if digest(AUDIT/f'{tag}_manifest.json')!=summary['audit_manifest_sha256'][tag]:
            raise ValueError(f'Existing audit manifest changed: {tag}')
    _, current_ref = protected()
    leak = read(prep/'leakage.json')
    if current_ref != leak['protected_references']:
        raise ValueError('Protected reference set changed: refresh leakage audit before curation')
    rows = read(AUDIT/'OCR_Data_manifest.json')
    source_rows = {r['path']:r for r in rows}
    groups=defaultdict(list)
    for g in read(prep/'glyph_review_manifest.json'):
        groups[g['parent']].append(g)
    out.mkdir(parents=True,exist_ok=False)
    glyphs=[]; parents=[]; classes=defaultdict(Counter); domains=Counter(); counts=Counter(); flags_all=Counter(); tiles=[]; sample_ids=[]
    for path, r in sorted(source_rows.items()):
        gs=groups.get(path,[])
        quarantine=leak['quarantined'].get(path,[])
        if not gs:
            parents.append(dict(parent=path,domain='Invalid',tier='D',reasons=[r['annotation']],
                                quarantine=quarantine,eligible={t:False for t in TASKS})); domains['Invalid']+=1
            continue
        if r['annotation']!='ok' or len(gs)!=len(r['boxes']): raise ValueError(f'Manifest member mismatch: {path}')
        p = safe_path(ext/'OCR_Data',path)
        label = p.parent.parent.parent/'labels'/p.parent.name/p.with_suffix('.txt').name
        if digest(p)!=r['sha256'] or digest(label)!=r['label_sha256']:
            raise ValueError(f'Extracted OCR file changed: {path}')
        im=load_image(p)
        if im.shape[:2]!=(r['height'],r['width']): raise ValueError(f'Dimensions changed: {path}')
        # Unknown domain is deliberately not promoted to real Vietnamese; large charts are only suspects.
        domain='Ambiguous'
        cues=['chart_or_multiple_regions_candidate'] if len(gs)>10 else []
        domains[domain]+=1
        tiers=Counter()
        for idx,g in enumerate(gs):
            target,x,y,w,h = r['boxes'][idx]
            box=[x,y,w,h]
            if (g['sha256'],g['label_sha256'],g['target_id'],g['character'],g['box_xywh'],g['quarantine']) != (
                r['sha256'],r['label_sha256'],forward[target],CLASSES[forward[target]],box,quarantine):
                raise ValueError(f'Glyph identity mismatch: {path} #{idx}')
            info, flags = glyph_flags(im, box, [b[1:] for j,b in enumerate(r['boxes']) if j!=idx])
            tier=classify_tier(flags,quarantine)
            tiers[tier]+=1; flags_all.update(flags); counts[tier]+=1
            if tier=='B': classes[g['original_split']][g['character']]+=1
            glyphs.append({**g,'glyph_index':idx,'domain':domain,'tier':tier,'pixel_diagnostics':info,
                           'glyph_flags':flags,'classifier_technical_candidate':tier=='B', 'eligible':{t:False for t in TASKS},
                           'classifier_missing':['semantic_glyph_evidence','training_rights','source_group_and_frozen_split']})
        parent_flags=gs[0]['quality_flags']+cues
        candidate=not quarantine and not parent_flags and tiers['B']==len(gs)
        parents.append(dict(parent=path,sha256=r['sha256'],domain=domain,tier='D' if quarantine else 'C' if not candidate else 'B',
                            original_split=r['split'],source_group=None,possible_duplicate_component=gs[0]['possible_duplicate_component'],
                            quarantine=quarantine,glyph_tiers=dict(tiers),quality_flags=parent_flags,
                            detector_review_candidate=candidate,completeness='UNKNOWN',verified_text=None,
                            eligible={t:False for t in TASKS}))
        if not quarantine and (cues or (parent_flags and tiers['B'] and not candidate)) and len(tiles)<20:
            sample_ids.append(path)
            tiles.append((f'{len(tiles)+1:02d} classifier B={tiers["B"]} / {len(gs)}',
                          draw(im.copy(),[[g['character'],*g['box_xywh']] for g in gs],(70,255,100))))
    write(out/'glyphs.json',glyphs); write(out/'parents.json',parents)
    sheet(out/'task_review.jpg',tiles); write(out/'task_review_index.json',sample_ids)
    vn=observe_vnlp(ext/'VNLP_detection',read(AUDIT/'VNLP_detection_manifest.json'),out)
    write(out/'summary.json',dict(glyph_tiers=counts,domains=domains,glyph_flags=flags_all,
        classifier_candidates_by_class={s:{c:v[c] for c in CLASSES} for s,v in classes.items()},
        glyph_candidates_in_detector_blocked_parents=sum(p.get('glyph_tiers',{}).get('B',0) for p in parents if not p.get('detector_review_candidate')),
        detector_review_candidates=sum(bool(p.get('detector_review_candidate')) for p in parents),
        verified_complete_character_images=0,verified_full_text_images=0,eligible={t:0 for t in TASKS},
        quarantine_images=sum(bool(p['quarantine']) for p in parents),protected_refs=len(current_ref),vnlp=vn,
        source_artifact_sha256=PINNED, rights='UNKNOWN_NO_DOCUMENTED_AUTHORIZATION'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,required=True); ap.add_argument('--extracted',type=Path,required=True)
    args=ap.parse_args()
    out=ROOT/'reports'/('task_curation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    curate(args.prepared.resolve(),args.extracted.resolve(),out)
    print(out)
"""Preflight and train OCR classifiers on an explicitly approved, frozen external cohort."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from external_task_curation import CLASSES, crop, digest, load_image, read, safe_path, write
from final_ocr_dataset import check_protocol


def evidence(root, receipt):
    return bool(isinstance(receipt, dict) and receipt.get('sha256') and receipt.get('path')
                and digest(safe_path(root,receipt['path']))==receipt['sha256'])


def preflight(curation, approval_path):
    """Validate evidence linkage; authenticity/rights must be checked by the approver."""
    if approval_path is None:
        raise ValueError('BLOCKED: no approval with training rights, glyph evidence and frozen splits')
    approval=read(approval_path); base=approval_path.parent
    if approval.get('purpose')!='academic_classifier_training' or not str(approval.get('approved_by') or '').strip():
        raise ValueError('Missing accountable approval or classifier training scope')
    for field in ('rights_evidence','source_group_evidence','semantic_evidence','leakage_evidence'):
        if not evidence(base,approval.get(field)):
            raise ValueError(f'Missing or changed {field}')
    if digest(curation/'summary.json')!=approval.get('curation_summary_sha256') or digest(curation/'glyphs.json')!=approval.get('curation_glyphs_sha256'):
        raise ValueError('Curation artifacts do not match approval')
    frozen=approval.get('frozen_manifest')
    if not evidence(base,frozen): raise ValueError('Frozen manifest missing or changed')
    manifest=read(safe_path(base,frozen['path']))
    by_id={(g['parent'],g['glyph_index']):g for g in read(curation/'glyphs.json')}
    result=[]; seen=set(); relations={}; splits=Counter()
    for r in manifest:
        key=(r['parent'],r['glyph_index']); g=by_id.get(key); split=r['split']
        group=str(r.get('source_group') or '').strip().casefold()
        if not g or key in seen or g['quarantine'] or g['tier']=='D' or r['sha256']!=g['sha256'] or r['character']!=g['character']:
            raise ValueError(f'Invalid, altered, quarantined or duplicate glyph {key}')
        if r.get('glyph_verified') is not True or not group or group=='unknown' or split not in ('train','development','test'):
            raise ValueError(f'Missing verification, split or group: {key}')
        for relation in [('group',group),('parent',key[0]),('hash',g['sha256']),('component',g['possible_duplicate_component'])]:
            if relations.setdefault(relation,split)!=split: raise ValueError(f'Cross-split overlap: {relation}')
        seen.add(key); splits[split]+=1
        result.append({**g,'split':split,'source_group':group})
    if min(splits[s] for s in ('train','development','test')) < 1:
        raise ValueError('All frozen splits must be nonempty')
    return result, frozen['sha256']


def train(rows, ext, output, backend, frozen_hash):
    from dataset import extract_hog_features
    from yolo_character_pipeline import normalize_glyph
    from sklearn.metrics import classification_report, confusion_matrix

    x={s:[] for s in ('train','development')}; y={s:[] for s in x}; refs={s:[] for s in x}
    parent=None; image=None
    for r in sorted(rows,key=lambda row:row['parent']):
        split=r['split']
        if split=='test': continue  # No pixel decode, feature extraction or evaluation on test.
        if parent!=r['parent']:
            parent=r['parent']
            path=safe_path(ext/'OCR_Data',parent)
            if digest(path)!=r['sha256']: raise ValueError(f'Image hash mismatch: {parent}')
            image=load_image(path)
        x[split].append(normalize_glyph(crop(image,r['pixel_xywh'])[0])); y[split].append(r['target_id'])
        refs[split].append([parent,r['glyph_index']])
    if len(set(y['train']))<2: raise ValueError('Need at least two training classes')
    output.mkdir(parents=True,exist_ok=False)
    if backend=='cnn':
        import torch
        from torch import nn
        from ocr_v3_experiment import network
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback disabled')
        torch.manual_seed(42)
        model=network().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
        scaler=torch.amp.GradScaler('cuda')
        xt={s:torch.tensor(np.stack(a),device='cuda',dtype=torch.float32)[:,None]/255 for s,a in x.items()}
        yt={s:torch.tensor(a,device='cuda',dtype=torch.long) for s,a in y.items()}
        criterion=nn.CrossEntropyLoss(); best=float('inf'); waiting=0; history=[]
        for epoch in range(60):
            model.train()
            total=0.
            for ids in torch.randperm(len(yt['train']),device='cuda').split(32):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    loss=criterion(model(xt['train'][ids]),yt['train'][ids])
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total+=loss.item()*len(ids)
            model.eval()
            with torch.inference_mode():
                logits=torch.cat([model(a) for a in xt['development'].split(128)])
                val=criterion(logits,yt['development']).item()
            scheduler.step(val); history.append([epoch+1,total/len(yt['train']),val])
            if val < best:
                best=val; waiting=0; pred=logits.argmax(1).cpu().numpy()
                torch.save(model.state_dict(),output/'cnn.pt')
            else:
                waiting+=1
                if waiting>=10: break
        write(output/'history.json',history)
    else:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        model=RandomForestClassifier(n_estimators=200,max_depth=24,n_jobs=4,random_state=42) if backend=='rf' else make_pipeline(StandardScaler(),SVC(kernel=backend.removeprefix('svm-'),C=3))
        hog={s:np.stack([extract_hog_features(a) for a in ims]) for s,ims in x.items()}
        model.fit(hog['train'],y['train']);pred=model.predict(hog['development'])
        joblib.dump(model,output/f'{backend}.joblib',compress=3)
    metrics=classification_report(y['development'],pred,labels=list(range(36)),target_names=list(CLASSES),zero_division=0,output_dict=True)
    write(output/'development.json',dict(backend=backend,seed=42,frozen_sha256=frozen_hash,classes=CLASSES,
          preprocessing='V1 32x32 uint8; CNN divide by 255 once; HOG 324',
          train_counts=dict(Counter(y['train'])),accuracy=float(np.mean(np.asarray(y['development'])==pred)),
          confusion_matrix=confusion_matrix(y['development'],pred,labels=list(range(36))).tolist(),per_class=metrics,
          predictions=[[*r,int(p)] for r,p in zip(refs['development'],pred)]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('curation',type=Path); ap.add_argument('--approval',type=Path)
    ap.add_argument('--fit',choices=('cnn','svm-linear','svm-rbf','rf'))
    ap.add_argument('--extracted',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    try:
        check_protocol()
        rows, sha=preflight(args.curation,args.approval)
        print(dict(status='PREFLIGHT_PASS',splits=dict(Counter(r['split'] for r in rows)),frozen_sha256=sha))
        if args.fit:
            if not args.extracted or not args.output: raise ValueError('--fit requires --extracted and --output')
            train(rows,args.extracted,args.output,args.fit,sha)
    except (ValueError,OSError,KeyError) as e:
        ap.exit(2,f'{e}\n')


if __name__=='__main__': main()
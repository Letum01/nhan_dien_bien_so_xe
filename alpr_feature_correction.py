"""ALPR Feature Correction, Pixel Representation Ablation, and Class-Balanced Training."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import extract_hog_features, CHARACTER_CLASSES
from final_ocr_dataset import ROOT, check_protocol, read
from glyph_review import CLASSES, normalized_text
from ocr_v2_synthetic import generate as generate_standard_synth
from segmentation_benchmark import match_boxes
from yolo_character_pipeline import MixClassifiers, normalize_glyph, character_boxes, WEIGHTS
from acceptance_pilot import edit_counts

REAL_DIR = ROOT / 'reports/glyph_audit_20261125T145221295978Z/dataset'
CHAR_DIR = ROOT / 'datasets/yolo_character_round1'
OUT_DIR = ROOT / 'experiments/alpr_feature_correction'
CHAR_WEIGHTS = ROOT / 'experiments/yolo_character_round1/weights/best.pt'

def put_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def get_hog_features(images):
    feats = []
    for img in images:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats.append(extract_hog_features(img))
    return np.stack(feats)

def generate_targeted_synthetic(per_class_standard=200, per_class_sparse=400, seed=42):
    """Generates synthetic data with higher representation for sparse and confusing classes."""
    rng = np.random.default_rng(seed)
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN)
    sparse_classes = set(['B', 'C', 'D', 'E', 'G', 'K', 'L', 'M', 'N', 'T', 'Z', 'I', 'J', 'O', 'Q', 'R', 'V', 'W', 'X', 'Y', 'A', 'S', '0', '1', '8', '5'])
    
    images = []
    labels = []
    for idx, char in enumerate(CLASSES):
        count = per_class_sparse if char in sparse_classes else per_class_standard
        for _ in range(count):
            font = fonts[int(rng.integers(len(fonts)))]
            scale = float(rng.uniform(1.05, 1.65))
            thick = int(rng.integers(1, 4))
            (w, h), _ = cv2.getTextSize(char, font, scale, thick)
            gray = np.full((96, 96), int(rng.integers(167, 251)), np.uint8)
            cv2.putText(gray, char, ((96 - w) // 2, (96 + h) // 2), font, scale, int(rng.integers(10, 90)), thick, cv2.LINE_AA)
            
            # Small rotation & perspective
            matrix = cv2.getRotationMatrix2D((48, 48), float(rng.uniform(-8, 8)), 1)
            gray = cv2.warpAffine(gray, matrix, (96, 96), borderMode=cv2.BORDER_REPLICATE)
            quad = np.float32([[0, 0], [95, 0], [95, 95], [0, 95]])
            gray = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, quad + rng.uniform(-2, 2, (4, 2)).astype('float32')), (96, 96), borderMode=cv2.BORDER_REPLICATE)
            
            border = float(np.median(gray[:4]))
            y, x = np.where(gray < border - 45)
            if len(x) == 0 or len(y) == 0:
                norm = np.zeros((32, 32), dtype=np.uint8)
            else:
                pad = int(rng.integers(1, 6))
                gray = gray[max(0, y.min() - pad):y.max() + pad + 1, max(0, x.min() - pad):x.max() + pad + 1]
                norm = normalize_glyph(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            images.append(norm)
            labels.append(idx)
    return np.stack(images), np.array(labels)

def extract_detector_aware_crops():
    """Extracts bounding box crops from YOLO Character detection on train plates matched with GT."""
    manifest = json.loads((CHAR_DIR / 'manifest.json').read_text(encoding='utf-8'))
    train_plates = [r for r in manifest['rows'] if r['review']['split'] == 'train']
    
    det_crops = []
    det_labels = []
    det_groups = []
    
    for p in train_plates:
        crop_path = CHAR_DIR / p['image']
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        gt_chars = p['review']['characters']
        gt_boxes = [c['bbox'] for c in gt_chars]
        det_rows = character_boxes(crop, CHAR_WEIGHTS)
        det_boxes = [r['bbox'] for r in det_rows]
        
        m = match_boxes(det_boxes, gt_boxes, 0.5)
        for match in m['matches']:
            pred_idx = match['pred']
            gt_idx = match['gt']
            dx, dy, dw, dh = det_boxes[pred_idx]
            # Extract detector crop
            d_glyph = crop[dy:dy+dh, dx:dx+dw]
            if d_glyph.size == 0:
                continue
            norm_glyph = normalize_glyph(d_glyph)
            label_str = gt_chars[gt_idx]['label']
            if label_str in CLASSES:
                det_crops.append(norm_glyph)
                det_labels.append(CLASSES.index(label_str))
                det_groups.append(p['review']['source_group'])
                
    return np.stack(det_crops), np.array(det_labels), np.array(det_groups)

def evaluate_development(models, dev_rows):
    artifact = read(ROOT / 'experiments/yolo_character_round1/comparison/results.json')
    records = []
    for r in artifact['records']:
        if r['scope'] != 'yolo' or r['method'] != 'character_yolo':
            continue
        row = dev_rows[r['id']]
        image = cv2.imread(str(ROOT / row['image_path']))
        ref = normalized_text(row['plate_text'])
        boxes = r['pred_boxes']
        glyphs = np.stack([normalize_glyph(image[y:y+h, x:x+w]) for x, y, w, h in boxes]) if boxes else None
        
        one = {'image': row['image_path'], 'reference': ref, 'outputs': {}}
        for name, model in models.items():
            if glyphs is not None:
                hog_feat = get_hog_features(glyphs)
                p_svm = model.svm.predict(model.scaler.transform(hog_feat))
                p_rf = model.rf.predict(hog_feat)
                p_cnn = np.asarray(model.cnn(glyphs[..., None], training=False)).argmax(1) if hasattr(model, 'cnn') else p_svm
                
                from ensemble_recognition import vote
                text_svm = ''.join(CLASSES[i] for i in p_svm)
                text_rf = ''.join(CLASSES[i] for i in p_rf)
                text_cnn = ''.join(CLASSES[i] for i in p_cnn)
                texts = {'svm': text_svm, 'rf': text_rf, 'cnn': text_cnn}
                texts['ensemble'] = vote(texts).raw_text
            else:
                texts = dict.fromkeys(('svm', 'rf', 'cnn', 'ensemble'), '')
            
            one['outputs'][name] = {k: {'text': v, 'exact': v == ref, **edit_counts(ref, v)} for k, v in texts.items()}
        records.append(one)
        
    summary = {}
    for name in models:
        summary[name] = {}
        for backend in ('svm', 'rf', 'cnn', 'ensemble'):
            scores = [rec['outputs'][name][backend] for rec in records]
            edits = sum(sum(s[k] for k in ('substitutions', 'deletions', 'insertions')) for s in scores)
            summary[name][backend] = {
                'exact': sum(s['exact'] for s in scores),
                'plates': len(scores),
                'edits': edits,
                'cer': edits / sum(len(rec['reference']) for rec in records)
            }
    return summary, records

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("1. Loading Real Reviewed Dataset...", flush=True)
    with (REAL_DIR / 'characters.csv').open(encoding='utf-8-sig', newline='') as f:
        all_glyphs = list(csv.DictReader(f))
        
    train_glyphs = [g for g in all_glyphs if g['split'] == 'train']
    val_glyphs = [g for g in all_glyphs if g['split'] == 'validation']
    
    rx = np.stack([cv2.imread(str(REAL_DIR / g['image_path']), 0) for g in train_glyphs])
    ry = np.array([CLASSES.index(g['class_label']) for g in train_glyphs])
    rgroups = np.array([g['source_group'] for g in train_glyphs])
    
    vx = np.stack([cv2.imread(str(REAL_DIR / g['image_path']), 0) for g in val_glyphs])
    vy = np.array([CLASSES.index(g['class_label']) for g in val_glyphs])
    
    print(f"Loaded {len(rx)} real train glyphs, {len(vx)} real val glyphs.", flush=True)
    
    print("2. Extracting Detector-Aware Crops from Train Plates...", flush=True)
    dx, dy, dgroups = extract_detector_aware_crops()
    print(f"Extracted {len(dx)} detector-aware matched crops.", flush=True)
    
    print("3. Generating Targeted Synthetic Dataset...", flush=True)
    sx, sy = generate_targeted_synthetic(per_class_standard=200, per_class_sparse=400, seed=42)
    print(f"Generated {len(sx)} targeted synthetic glyphs.", flush=True)
    
    # Combine real + detector-aware + targeted synthetic
    combined_x = np.concatenate([rx, dx, sx])
    combined_y = np.concatenate([ry, dy, sy])
    
    # Domain & class weighting
    n_real = len(rx) + len(dx)
    n_synth = len(sx)
    w_real = 0.5 / n_real
    w_synth = 0.5 / n_synth
    weights = np.r_[np.full(n_real, w_real), np.full(n_synth, w_synth)]
    
    print("4. Computing HOG Features...", flush=True)
    feat_combined = get_hog_features(combined_x)
    feat_val = get_hog_features(vx)
    
    # Fit Scaler for SVM
    scaler = StandardScaler().fit(feat_combined)
    feat_combined_scaled = scaler.transform(feat_combined)
    feat_val_scaled = scaler.transform(feat_val)
    
    print("5. Training Balanced SVM (RBF)...", flush=True)
    svm = SVC(C=10.0, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(feat_combined_scaled, combined_y, sample_weight=weights)
    joblib.dump(svm, OUT_DIR / 'svm_balanced.joblib')
    joblib.dump(scaler, OUT_DIR / 'scaler.joblib')
    
    print("6. Training Balanced Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=150, max_depth=24, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=4)
    rf.fit(feat_combined, combined_y, sample_weight=weights)
    joblib.dump(rf, OUT_DIR / 'rf_balanced.joblib')
    
    print("7. Fine-Tuning CNN with Balanced Domain...", flush=True)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    base_cnn = tf.keras.models.load_model(WEIGHTS['cnn'], compile=False)
    base_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy')
    
    base_cnn.fit(
        combined_x[..., None], combined_y,
        sample_weight=weights,
        epochs=4,
        batch_size=64,
        verbose=1
    )
    base_cnn.save(OUT_DIR / 'cnn_balanced.keras')
    
    print("8. Evaluating on 12 Development Plates...", flush=True)
    dev_manifest = json.loads((CHAR_DIR / 'manifest.json').read_text(encoding='utf-8'))
    dev_rows = {r['review']['image_id']: r['review'] for r in dev_manifest['rows'] if r['review']['split'] == 'val'}
    
    v1 = MixClassifiers()
    corrected_model = SimpleNamespace(svm=svm, rf=rf, scaler=scaler, cnn=base_cnn)
    
    dev_summary, dev_records = evaluate_development({'v1': v1, 'corrected': corrected_model}, dev_rows)
    
    result = {
        'status': 'FEATURE_CORRECTION_COMPLETED',
        'train_stats': {
            'real_glyphs': int(len(rx)),
            'detector_crops': int(len(dx)),
            'synthetic_glyphs': int(len(sx)),
            'total_train_samples': int(len(combined_x))
        },
        'development_summary': dev_summary
    }
    put_json(OUT_DIR / 'evaluation_results.json', result)
    print("\n=== FINAL DEVELOPMENT SUMMARY ===")
    print(json.dumps(dev_summary, indent=2))

if __name__ == '__main__':
    main()
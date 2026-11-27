"""Opt-in demo heads; leave the frozen detector and OCR pipeline unchanged."""
from collections import Counter
from time import perf_counter

import numpy as np

from dataset import extract_hog_features
from dual_yolo_pipeline import run
from final_ocr_dataset import ROOT, check_protocol, read, sha
from yolo_character_pipeline import CLASSES, normalize_glyph, segment

FUSION = ('svm_fusion', 'rf_fusion')


class FusionHeads:
    def __init__(self, cnn):
        import joblib
        import tensorflow as tf

        check_protocol()
        expected = read(ROOT/'reports/adaptive_score_probe.json')['head_hashes']
        self.heads = {}
        for name in FUSION:
            path = ROOT/'experiments/hybrid_features_01'/f'{name}.joblib'
            if sha(path) != expected[name]:
                raise ValueError(f'Checkpoint Fusion không khớp: {name}')
            head = joblib.load(path)
            if (head['classes'] != list(CLASSES) or list(head['classifier'].classes_) != list(range(36))
                    or head['classifier'].n_features_in_ != 580
                    or (name == 'svm_fusion' and head['scaler'].n_features_in_ != 580)):
                raise ValueError(f'Sai hợp đồng đặc trưng Fusion: {name}')
            self.heads[name] = head
        inner = cnn.layers[-1]
        self.embed = tf.keras.Model(inner.inputs, inner.get_layer('dense').output)

    def predict(self, glyphs, names, timings):
        start = perf_counter()
        pixels = np.stack(glyphs)
        hog = np.stack([extract_hog_features(g) for g in pixels])
        embedding = np.asarray(self.embed([pixels[..., None]], training=False))
        features = np.concatenate((hog, embedding), axis=1)
        if features.shape != (len(glyphs), 580) or not np.isfinite(features).all():
            raise ValueError('Đặc trưng Fusion không hợp lệ')
        timings['fusion_features'] = perf_counter()-start
        texts = {}
        for name in names:
            start = perf_counter()
            h = self.heads[name]
            x = h['scaler'].transform(features) if name == 'svm_fusion' else features
            texts[name] = ''.join(CLASSES[int(i)] for i in h['classifier'].predict(x))
            timings[name] = perf_counter()-start
        return texts


def run_fusion(image, worker, models, get_heads, backend, method):
    start = perf_counter()
    compare = backend == 'compare'
    result = run(image, worker, models, 'all' if compare else 'cnn', method)
    names = FUSION if compare else (backend,)
    texts = dict.fromkeys(names, '')
    if result['glyph_boxes']:
        x,y,w,h = result['plate']['crop_box']
        crop = image[y:y+h,x:x+w]
        if method == 'character_yolo':
            glyphs = [normalize_glyph(crop[b:b+d,a:a+c]) for a,b,c,d in result['glyph_boxes']]
        else:
            # Replay the existing deterministic OpenCV pixel path, not raw box crops.
            rows = segment(crop, method)
            if [list(box) for _,box in rows] != result['glyph_boxes']:
                raise ValueError('Fusion không khớp thứ tự box OpenCV')
            glyphs = [g for g,_ in rows]
        texts = get_heads().predict(glyphs, names, result['timings'])
    result['text'] = {**result['text'],**texts} if compare else texts
    result['timings']['total'] = perf_counter()-start
    return result


def agreement(texts):
    """Counts only a comparable trio on the same ordered glyphs."""
    names = ('cnn', *FUSION) if all(k in texts for k in FUSION) else ('cnn','svm','rf')
    if not all(k in texts for k in names): return [], ()
    strings = [texts[k] for k in names]
    if len({len(s) for s in strings}) != 1: return [], ()
    return [max(Counter(s).values()) for s in zip(*strings)], names
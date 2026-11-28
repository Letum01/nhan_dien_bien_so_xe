"""Opt-in demo heads; leave the frozen detector and OCR pipeline unchanged."""
from collections import Counter
from time import perf_counter

import numpy as np

from dataset import extract_hog_features
from dual_yolo_pipeline import run
from final_ocr_dataset import ROOT, read, sha
from runtime_validation import validate_runtime, validate_fusion_head
from yolo_character_pipeline import CLASSES, WEIGHTS

FUSION = ('svm_fusion', 'rf_fusion')


def load_cnn():
    """Load only the verified CNN, not the historical HOG-only heads."""
    import tensorflow as tf

    validate_runtime()
    path = WEIGHTS['cnn']
    model = tf.keras.models.load_model(path, compile=False)
    def rescalers(layer):
        if isinstance(layer, tf.keras.layers.Rescaling):
            yield layer
        for child in getattr(layer, 'layers', ()):
            yield from rescalers(child)
    scales = list(rescalers(model))
    if (model.input_shape != (None, 32, 32, 1) or model.output_shape != (None, 36)
            or read(path.parent/'cnn_contract.json')['class_order'] != list(CLASSES)
            or len(scales) != 1 or not np.isclose(scales[0].scale, 1/255) or scales[0].offset != 0):
        raise ValueError('CNN input/output/rescaling mismatch')
    return model


class FusionHeads:
    def __init__(self, cnn):
        import joblib
        import tensorflow as tf

        spec = validate_runtime()
        self.heads = {}
        for name in FUSION:
            path = ROOT/'experiments/hybrid_features_01'/f'{name}.joblib'
            if sha(path) != spec['artifacts'][path.relative_to(ROOT).as_posix()]['sha256']:
                raise ValueError(f'Checkpoint Fusion không khớp: {name}')
            head = joblib.load(path)
            validate_fusion_head(head, name)
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


def run_fusion(image, worker, cnn, get_heads, backend):
    """Capture the exact normalized batch at the frozen pipeline boundary."""
    from types import SimpleNamespace

    start = perf_counter()
    compare = backend == 'compare'
    pixels = None

    def capture(batch, *, training=False):
        nonlocal pixels
        pixels = batch[..., 0]
        return cnn(batch, training=training)

    result = run(image, worker, SimpleNamespace(cnn=capture), 'cnn')
    names = FUSION if compare else (backend,)
    texts = dict.fromkeys(names, '')
    if pixels is not None:
        texts = get_heads().predict(pixels, names, result['timings'])
    result['text'] = {**result['text'],**texts} if compare else texts
    result['timings']['total'] = perf_counter()-start
    return result


def agreement(texts):
    """Counts only a comparable trio on the same ordered glyphs."""
    names = ('cnn', *FUSION)
    if not all(k in texts for k in names): return [], ()
    strings = [texts[k] for k in names]
    if len({len(s) for s in strings}) != 1: return [], ()
    return [max(Counter(s).values()) for s in zip(*strings)], names

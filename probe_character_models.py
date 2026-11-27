"""Known GENERATED labels test plumbing only, never real glyph accuracy."""
import json
from pathlib import Path

import numpy as np

from dataset import generate_synthetic_character_dataset,CHARACTER_CLASSES,extract_hog_features
from main import load_trained_models

ROOT=Path(__file__).resolve().parent


def main():
    models=load_trained_models(str(ROOT/'saved_models'))
    images,labels=generate_synthetic_character_dataset(samples_per_class=3)
    features=np.array([extract_hog_features(image.squeeze()) for image in images],np.float32)
    predictions={'svm':models['svm'].predict(models['scaler'].transform(features)),
                 'rf':models['rf'].predict(features),
                 'cnn':models['cnn'].predict(images,verbose=0).argmax(axis=1)}
    from collections import Counter
    output=ROOT/'reports/ocr_diagnostics';output.mkdir(exist_ok=True)
    report={'scope':'Generated Hershey font probes, possible training-distribution overlap. NOT real OCR benchmark. No real data accuracy.',
            'samples':len(labels),'mapping':CHARACTER_CLASSES,'results':{}}
    for model,predicted in predictions.items():
        report['results'][model]={'generated_label_matches':int((predicted==labels).sum()),
            'predicted_class_counts':dict(Counter(CHARACTER_CLASSES[int(i)] for i in predicted)),
            'expected':''.join(CHARACTER_CLASSES[int(i)] for i in labels),
            'predicted':''.join(CHARACTER_CLASSES[int(i)] for i in predicted)}
    (output/'synthetic_plumbing_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v['generated_label_matches'] for k,v in report['results'].items()},'out of',len(labels),'GENERATED ONLY')


if __name__=='__main__':main()
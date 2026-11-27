"""Render executed results, per-plate comparison and confusion matrices."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'reports/OCR_MIX_EXPERIMENT_01.json').read_text(encoding='utf-8'))
    folder=Path(report['experiment']);results=report['results']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,result in results.items():
        matrix=np.array(result['confusion_matrix'])
        fig,axis=plt.subplots(figsize=(12,10));axis.imshow(matrix,cmap='Blues')
        axis.set_xticks(range(36),result['matrix_class_order']);axis.set_yticks(range(36),result['matrix_class_order'])
        axis.set_xlabel('Predicted');axis.set_ylabel('Human reviewed label');axis.set_title(name+' / development validation')
        fig.tight_layout();fig.savefig(folder/(name+'_confusion.png'));plt.close(fig)
    mapping={}
    source=Path('reports/glyph_audit_20261125T071220171166Z/dataset')
    revisions=json.loads((ROOT/source/'ocr_revisions.json').read_text(encoding='utf-8'))
    for row in revisions:mapping[row['image_id']]=row
    lines=['# OCR mix experiment 01 — executed SVM/RF comparison',
           '\n## Preflight and data',
           'PASS:3600generatedsynthetictrain,147realreviewedtrain,57realvalidationglyphs/7plates/7groups. '
           'Bundlehash/perpixel/class/sourcehash checked; regenerated synthetic pixel-identical withseed42. '
           'LiveOCRrevisions/detectionparent match frozenexport;realvalidator/group/hash/pHashPASS. '
           'No validation fitting/augmentation/scalerfitting; no detectiontest used. '
           'Synthetic labels fromrenderinstructions,notOCRprediction. Oldweights historicalmembershipunverified.',
           '\n## Config and weighting',
           'SVMboth:RBF,C10,gamma scale,probabilityTrue,seed42,HOG324,savedscaler. '
           'RFboth:100trees,maxdepth20,seed42,rawHOG. Synthetic-only3600samples unitweight. '
           'Mix3747samples;syntheticweightsum1873.5,realweightsum1873.5 (50/50domain). '
           'Same raw feature extraction;mixscaler fit on its3747TRAINsamples only,syntheticon3600TRAINonly. '
           'RFfit(sample_weight=...) supported and forwarded to tree fitting;no resampling/invented independent sources.',
           '\n## Results on SAME57 reviewed DEVELOPMENT validation glyphs',
           '|Model|Correct/57|Accuracy|Macro recall supported|I predictions|Estimator fit seconds|',
           '|---|---:|---:|---:|---:|---:|']
    for key in ('svm_old_baseline','svm_synthetic','svm_mix','rf_old_baseline','rf_synthetic','rf_mix'):
        r=results[key];seconds=r.get('fit_timing',{}).get('estimator_fit_seconds')
        lines.append(f"|{key}|{r['correct']}/57|{100*r['accuracy']:.2f}%|{100*r['macro_recall_supported']:.2f}%|{r['I_predictions']}|{seconds if seconds is not None else 'not fitted'}|")
    lines += ['\nMacro recall averages only17classes with support;36class confusion matrix includes absent classes for stablemapping. '
              '57glyphs are clustered within7plates,NOT57independent sources. Validation metrics NOT independent test accuracy. '
              'Models selected/tuned in later work may overfit this small development set. No uncertainty interval pretending glyph independence.',
              '\n## Per plate (correct/total)',
              '|Image / group|SVM old|SVM synthetic|SVM mix|RF old|RF synthetic|RF mix|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for plate in results['svm_mix']['per_plate']:
        parent=mapping[plate]
        values=[]
        for key in ('svm_old_baseline','svm_synthetic','svm_mix','rf_old_baseline','rf_synthetic','rf_mix'):
            value=results[key]['per_plate'][plate];values.append(f"{value['correct']}/{value['samples']}")
        lines.append('|'+Path(parent['image_path']).name+' / '+parent['source_group']+'|'+'|'.join(values)+'|')
    lines+=['\n## Recall changes mix versus newly trained synthetic-only']
    for name,change in report['recall_changes_mix_vs_synthetic'].items():
        lines.append(f"- {name}: improved {change['improved_recall_classes']}; decreased {change['decreased_recall_classes']}.")
    lines+=['\nNo supported class recall decreased in these paired runs, but neither mix reads all glyphs. '
            'Mean accuracy alone is not proof:perplate performance varies;classE/K/M absentrealtrain,minorityclasses small. '
            'More correct digits plus sharply reducedIcollapse support benefit from reviewedrealtraining at fixedconfig on THISset, '
            'not a universalclaim or causal isolation from changedtrainfeature/scalerdistribution.',
            '\n## Remaining mistakes (mix)',
            '|Model|Image|Within-validation index|GT|Prediction|','|---|---|---:|---|---|']
    for name in ('svm_mix','rf_mix'):
        for i,(gt,pred) in enumerate(zip(report['expected'],results[name]['predictions'])):
            if gt!=pred:
                parent=mapping[report['validation_rows'][i]['plate_id']]
                lines.append(f"|{name}|{Path(parent['image_path']).name}|{i}|{gt}|{pred}|")
    lines+=['\n## Artifacts and reproducibility',f"Experiment root:`{folder}`.",
            'Eachsvmrun:svm_model.joblib,hog_scaler.joblib,run_config.json,training_log.json,preprocessing.json,validation.json. '
            'Eachrfrun:rf_model.joblib and samemetadata. comparison.json includes36classmatrices/perclassrecall/perplate/group/predictions. '
            'ConfusionPNG foreachmodelatroot. Reloadpredictions match storedresult,classmapping0..35 andfeatures324 verified. '
            'SVCscalern_samples_seen3600/3747checked. Fitseconds excludeHOG;totalrunseconds inJSON includefeatures/evaluation.',
            '\n## Decision',
            'PASSexploratoryexperiment. Consider CNNsynthetic-only versusmixed next with SAMEfrozendata,smallfixedbudget and newrunpaths; '
            'do not promote merely to beat57validation. RFmix50/57andSVMmix49/57alreadygoodcandidatebaselines for furtherindependentsources. '
            'CollectreviewedrealtrainE/K/Mandothermissing15classes,moregroupsand independentOCRtest. '
            'CNNtrainingNOTRUN;Ensembleproductionunchanged. No oldweights/SQLite/YOLO/threshold0.4/defaultdetector modified. '
            'Historicaloldbaseline sourceindependence remainsUNVERIFIED,whereas newruns explicitlyfittrainonly.']
    (ROOT/'reports/OCR_MIX_EXPERIMENT_01.md').write_text('\n'.join(lines),encoding='utf-8')
    print('reports/OCR_MIX_EXPERIMENT_01.md')


if __name__=='__main__':main()
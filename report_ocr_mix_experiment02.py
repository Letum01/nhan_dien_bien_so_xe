"""Render completed CNN paired experiment; run in plotting-capable venv-yolo."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads((ROOT/'reports/OCR_MIX_EXPERIMENT_02.json').read_text(encoding='utf-8'))
    folder=Path(report['experiment']);histories={}
    for name in ('cnn_synthetic','cnn_mix'):
        with (folder/name/'training.csv').open() as stream:history=list(csv.DictReader(stream))
        histories[name]={'epochs':len(history),'best_epoch':int(min(history,key=lambda r:float(r['val_loss']))['epoch'])+1,
                         'last':history[-1],'best':min(history,key=lambda r:float(r['val_loss']))}
        plt.figure()
        for field in ('loss','val_loss'):plt.plot([int(r['epoch'])+1 for r in history],[float(r[field]) for r in history],label=field)
        plt.legend();plt.xlabel('Epoch');plt.title(name);plt.savefig(folder/name/'learning_curve.png');plt.close()
    for name,result in report['results'].items():
        plt.figure(figsize=(12,10));plt.imshow(result['confusion_matrix'],cmap='Blues')
        plt.xticks(range(36),result['matrix_class_order']);plt.yticks(range(36),result['matrix_class_order'])
        plt.xlabel('Prediction');plt.ylabel('Reviewed label');plt.title(name+' / development validation')
        plt.tight_layout();plt.savefig(folder/(name+'_confusion.png'));plt.close()
    lines=['# OCR mix experiment 02 — CNN execution report',
           '\n## Data/preflight',
           'PASS:3600synthetictrain,147realtrain,57reviewedvalidationglyphsfrom7plates/7groups. '
           'Generator regeneration/hash,parent/group/pHashchecksPASS. No validation fit/augmentation/scaler. '
           'No OCRtest. Existing syntheticbaseline historicaltrainprovenanceUNVERIFIED. '
           'This is development validation, NOT final independent test accuracy.',
           '\n## Training contract',
           'TensorFlow2.16.2 onCPU (nativeWindows GPUlistempty),seed42. BothrunssameinitialweightSHA, '
           'sameexistingCNNarchitecture,batch16,max30epochs,AdamLR.001,EarlyStoppingval_losspatience5, '
           'ReduceLROnPlateaupatience2/factor.5,ModelCheckpointbestval_loss. '
           'Trainonlyrotation±5deg/translation.03. Input32x32x1,0..255;exactly1internalRescaling1/255. '
           'Synthetic-only3600unitweights;mix3747samples,totaldomainweights1873.5each. '
           'Keras sample_weight multiplies perexampleLOSS;accuracy metric is unweighted,not domainbalanced. '
           'BN sees actual mixed batch frequencies,not sample-weight-balanced statistics;this is a limitation.',
           '\n|Run|Epochs actual|Best epoch|Restored best val loss|Seconds|',
           '|---|---:|---:|---:|---:|']
    for name,history in histories.items():
        lines.append(f"|{name}|{history['epochs']}|{history['best_epoch']}|{float(history['best']['val_loss']):.6f}|{report['results'][name]['total_seconds']:.2f}|")
    lines+=['\n## Same validation glyph comparison',
            '|Model|Correct/57|Accuracy|Macro recall supported17classes|I predictions|','|---|---:|---:|---:|---:|']
    for name,r in report['results'].items():
        lines.append(f"|{name}|{r['correct']}/57|{r['accuracy']:.4%}|{r['macro_recall_supported']:.4%}|{r['I_predictions']}|")
    lines+=['\nMajority usesSVMmix/RFmix/CNNmix,sameglyphorder,>=2votes;tie→?. '
            'It scores52/57,less thanCNNmix54/57;doNOTpromoteEnsembleasbetter. '
            'No weighted/soft voting or confidence calibration enabled.',
            '\nCNNmix mistakes (validation indices):2 D→0,11 A→N,35 M→N. '
            'ClassM hasno realtrainexample;D/A eachscarce. ClasseswithoutrealtrainE/K/M: '
            f"{report['results']['cnn_mix']['unseen_real_train_classes']}. Perclassrecall,matrix,perplate/group,allglyphpredictions storedinJSON.",
            '\n## Segmentation versus classification',
            '|Image|Reviewed glyphs|Auto components|CNNmix auto text|Majority auto text|','|---|---:|---:|---|---|']
    for r in report['segmentation_vs_reviewed']:
        lines.append(f"|{Path(r['image']).name}|{r['reviewed_glyphs']}|{r['automatic_glyphs']}|{r['automatic_predictions']['cnn']}|{r['ensemble_raw']}|")
    lines+=['\nAllsevenarehuman-reviewedGTplatecrops,notYOLOendtoend. Greenreviewedglyph/redautocontour overlays '
            'andpersegmentbestIoUincomparison.json. Countdifferenceprovescomponentsmismatch,not labelsforautoglyphs. '
            '14.jpg extranarrowframeglyph→prefix;0408extraverticaldetails;15.jpg remainsnoisy/disordered/mismatched. '
            'Equalcountin13.jpg/AQUA2doesnotprovealignment. No automaticglyphaccuracycomputedfromindex. '
            'CNNmix exacttextonGTcropagreementsare diagnostic only,not reportedasYOLOplateaccuracy. '
            'Prioritynow:segmentation filtering/row/order benchmark using reviewedboxoverlap,independent data,before endtoendpromotion.',
            '\n## Technical issues and verification',
            'Initial runner evaluation failed converting stringpredictiontoNumpy scalar. metricshelpernow convertslist; '
            'synthetictrainedartifactreusedwithoutrefit/overwrite viaexplicitresume. '
            'NestedKerasoptimizerloadwarnings observed;compile=Falseinference reload matchesallstoredpredictions; '
            'inferenceweights/shape verified. These warnings do not establish resumable optimizerstate. '
            'TFretracing warningsduringvaryingglyphbatch smoke logged,not hidden.',
            '\nPreflight/postcheck verifies sourceSQLite revisions/drafts andbaselinehashesunchanged. '
            'No classifierproduct/defaultdetector/YOLO/threshold/split changes. Best/finalCNNcheckpoints retainedinrunfolders. '
            'Learningcurves/confusionplotscreated,newartifactpathsnooriginalweights overwritten.',
            '\n## Readiness',
            'Classificationdevelopmentimprovement supported54/57vs29/57atfixedprotocol,notgeneralizationclaim. '
            'OCRendtoendNOTREADY:segmentationerrors,15realtrainmissingclasses,7validationclusters,noindependentOCRtest. '
            'CollectrealD/A/Mandmissingclasses,newsourcegroups,testglyphs;avoid repeatedtuningon57glyphs. '
            'KeepCNNmix,SVMmix,RFmixasexperimentalcandidatesonly. No product model replacement.']
    report['training_history_summary']=histories
    (ROOT/'reports/OCR_MIX_EXPERIMENT_02.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/OCR_MIX_EXPERIMENT_02.md').write_text('\n'.join(lines),encoding='utf-8')
    print('reports/OCR_MIX_EXPERIMENT_02.md/json')


if __name__=='__main__':main()
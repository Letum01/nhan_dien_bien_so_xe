"""Consolidate executed split-B artifacts into a reproducible final report."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    experiment=ROOT/'experiments/yolo_round2_splitB_run01'
    migration=ROOT/'reports/splitB_applied_20261124T154151235536Z'
    receipt=json.loads((migration/'receipt.json').read_text(encoding='utf-8'))
    runtime=json.loads((experiment/'pilot_report.json').read_text(encoding='utf-8'))
    comparison=json.loads((experiment/'validation_comparison/comparison.json').read_text(encoding='utf-8'))
    with (experiment/'results.csv').open() as stream:history=list(csv.DictReader(stream))
    report={'dataset':{'reviewed':61,'before':{'train':17,'val':41,'test':3},'after':receipt['after_counts'],
                       'migration_receipt':str(migration/'receipt.json'),'backup':receipt['backup'],
                       'changes':receipt['changes'],'source_groups_after':{'train':40,'val':12,'test':3},
                       'pilot_invariant':'PASS','historical_revisions':'PASS','idempotent_retry':'PASS',
                       'quality_leakage_validator':'PASS','dataset':str(ROOT/'reports/round2_gate_20261124T154153249700Z/dataset')},
            'training':{**runtime,'status':'PASS','epochs_actual':len(history),'best_epoch':37,'early_stop_patience':20,
                        'epoch_last':history[-1],'epoch_best':history[36],
                        'best_sha256':hashlib.sha256((experiment/'weights/best.pt').read_bytes()).hexdigest(),
                        'initialization':'pretrained/yolov8n.pt, not pilot','test_evaluation':'NOT_TESTED_deferred'},
            'comparison':comparison,'integration':{'status':'PASS_technical_only','crops':12,'ocr_backend_errors':0,
                                                  'report':str(experiment/'integration_smoke/results.json'),
                                                  'default_detector_changed':False,'ocr_weights_changed':False},
            'tests':{'pytest_pass':96,'pip_check_main':'PASS','pip_check_yolo':'PASS','git_diff_check':'PASS'},
            'limits':['Only46 train and12 validation;3 test deferred, high uncertainty',
                      'Validation used for early stopping and checkpoint selection; not final unbiased estimate',
                      '33 formerly validation images now train; excluded from common validation comparison',
                      'No product threshold change, no automatic promotion']}
    (ROOT/'reports/SPLIT_B_FINAL.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Split B — báo cáo kết quả thực thi',
           '\n## Dataset / migration',
           'User clarified approval of exact33 changes:31 val→train and2 train→val.61 reviewed retained;17/41/3→46/12/3. '
           'All21 pilot payloads unchanged. New33 revisions, original history retained; backup integrity verified and idempotent retry produced no new revisions. '
           'Only split/annotation_version changed; original reviewed_at retained and saved_at records migration time.',
           f'\nBackup/receipt: `{migration}`. Snapshot quality/group/hash/pHash/export validator PASS.',
           '\n## Training',
           "CUDA RTX2060, torch2.5.1+cu121 / ultralytics8.3.40; pretrained original. Maximum100, early-stop at57; best37. "
           "Batch4,imgsz640,patience20,seed42,workers2,AMP. No hyperparameter tuning on test.",
           f"\nRuntime train call {runtime['training_seconds']:.2f}s; CSV loop {history[-1]['time']}s. Peak allocated {runtime['peak_cuda_allocated_bytes']} bytes, reserved {runtime['peak_cuda_reserved_bytes']} bytes (allocator, not total GPU memory).",
           f"\nFinal loss train box/cls/DFL: {history[-1]['train/box_loss']}/{history[-1]['train/cls_loss']}/{history[-1]['train/dfl_loss']}; val: {history[-1]['val/box_loss']}/{history[-1]['val/cls_loss']}/{history[-1]['val/dfl_loss']}.",
           f'\nBest: `{experiment / "weights/best.pt"}`; last.pt also load PASS. Test NOT evaluated.',
           '\n## Same validation comparison',
           '12 images excluded from both models train IDs/groups; source/hash leakage PASS. Same640 square preprocessing,NMS IoU.7,max_det300. '
           'Green GT/yellow pilot/red round2 overlays under validation_comparison/. The new model selected on this validation, so improvement is developmental, not generalization proof.',
           '\n|Model|Threshold|TP|FP|FN|Precision|Recall|TP confidence|TP IoU|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison['threshold_summary']:
        def value(v):return 'N/A' if v is None else f'{v:.4f}'
        lines.append(f"|{r['model']}|{r['threshold']}|{r['tp']}|{r['fp']}|{r['fn']}|{value(r['precision'])}|{value(r['recall'])}|{value(r['mean_tp_confidence'])}|{value(r['mean_tp_iou'])}|")
    lines+=['\nUltralytics mAP on the SAME12val (rect=False):']
    for name,metrics in comparison['ultralytics_val_metrics'].items():lines.append(f'- {name}: {metrics}')
    lines+=['\nTraining-loop best validation metrics differ slightly due validation preprocessing/batching; do not mix these with the same-square comparison metrics.',
            '\nAt conf.4 round2:11TP/1FP/1FN; pilot:0TP/0FP/12FN. TP score round2~.905. '
            'Known failure AQUA2_20449_checkoutex_2020-10-30-16-10uF9O9vXoMl.jpg has both1FP/1FN. '
            'No threshold was selected from test.0.4 is a promising experimental operating point on this validation, NOT production readiness.',
            '\n## Integration / tests',
            'YOLO adapter produced12 crops at.4 on validation; segmentation and SVM/RF/CNN/Ensemble smoke ran without backend errors. '
            'No OCR accuracy claimed. No default detector/threshold/OCR checkpoint change. Product Streamlit AppTests included in96pytest PASS. '
            'Both pip checks and git diff check PASS. Browser live round2 checkpoint choice NOT TESTED; isolated venv inference used.',
            '\n## Conclusions',
            'Split migration PASS; pilot intact PASS; dataset validator PASS; training PASS with early stopping. '
            'Confidence and detection improved on common validation; FP at.01 reduced1219→162, but at.4 new model has1FP vs pilot0FP '
            '(pilot detects nothing). Do not claim all FP reduced at every threshold. Test evaluation deferred.',
            'Continue collecting80–100 additional independently sourced reviewed train images, negatives and hard cases; preserve the frozen test. '
            'Investigate failure box/crop quality on validation, then predeclare a new evaluation protocol before another experiment. '
            'Only perform final test after model/threshold decisions are frozen. No high-accuracy claim from61 images.']
    (ROOT/'reports/SPLIT_B_FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    print('reports/SPLIT_B_FINAL.md/json written')


if __name__=='__main__':main()
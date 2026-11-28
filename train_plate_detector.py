"""Explicit opt-in YOLO training/validation on human annotated plate boxes."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Human-reviewed YOLO dataset YAML with train/val/test")
    parser.add_argument("architecture", type=Path, help="Local YOLO architecture YAML; no automatic pretrained weight download")
    parser.add_argument("output", type=Path, help="New experiment directory outside saved_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrained", type=Path, help="Optional trusted local nano/small pretrained checkpoint; never auto-download")
    parser.add_argument("--device", default="cpu", help="Use 0 only after torch.cuda.is_available() is verified")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--config", type=Path, help="JSON training configuration")
    parser.add_argument("--skip-test",action='store_true',help='Defer test evaluation until selection is frozen; validation only')
    parser.add_argument('--preflight', action='store_true', help='Validate inputs only; never load or fit YOLO')
    parser.add_argument('--source-root', type=Path, help='Read-only parent-image artifact pack')
    args = parser.parse_args()
    protected = Path(__file__).resolve().parent / "saved_models"
    destination = args.output.resolve()
    if destination.exists() or destination == protected or protected in destination.parents:
        raise ValueError("Use a new experiment directory outside saved_models")
    if not args.data.is_file() or not args.architecture.is_file():
        raise FileNotFoundError("Local data YAML and architecture YAML required")
    from validate_yolo_dataset import validate_dataset
    validated = validate_dataset(args.data.resolve().parent, args.source_root)
    if args.preflight:
        print(validated)
        return
    from ultralytics import YOLO
    import yaml
    import json
    import torch
    settings=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    settings.pop("notes",None)
    device=str(settings.get("device",args.device))
    if device!="cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this PyTorch environment; no silent CPU fallback")
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if Path(config.get('path','')).resolve()!=args.data.resolve().parent:
        raise ValueError('Export a new dataset/YAML with its actual root before fitting; historical snapshots remain read-only')
    if config.get("names") not in ({0: "license_plate"}, ["license_plate"]) or not all(config.get(s) for s in ("train", "val", "test")):
        raise ValueError("Dataset must define license_plate only and separate train/val/test splits")
    if args.pretrained and not args.pretrained.is_file():
        raise FileNotFoundError("Pretrained checkpoint must be a trusted local file")
    model = YOLO(str(args.pretrained or args.architecture))
    settings={"epochs":args.epochs,"device":args.device,"batch":args.batch,**settings}
    import time
    training_start=time.perf_counter()
    if device!='cpu':
        torch.cuda.reset_peak_memory_stats()
    try:
        model.train(data=str(args.data.resolve()), pretrained=bool(args.pretrained), **settings,
                    project=str(destination.parent), name=destination.name, exist_ok=False)
    except torch.cuda.OutOfMemoryError:
        if int(settings.get('batch',4))<=4:
            raise
        import gc
        import warnings
        warnings.warn('CUDA OOM: retry from original initialization with batch=4 in separate _batch4 experiment')
        del model
        gc.collect();torch.cuda.empty_cache()
        destination=destination.with_name(destination.name+'_batch4')
        if destination.exists():raise FileExistsError(destination)
        settings['batch']=4
        model=YOLO(str(args.pretrained or args.architecture))
        model.train(data=str(args.data.resolve()),pretrained=bool(args.pretrained),**settings,
                    project=str(destination.parent),name=destination.name,exist_ok=False)
    trained = YOLO(str(destination / "weights" / "best.pt"))
    training_seconds=time.perf_counter()-training_start
    test=None if args.skip_test else trained.val(data=str(args.data.resolve()), split="test", project=str(destination), name="test_metrics",
                     device=device, batch=int(settings.get('batch',4)),workers=int(settings.get('workers',2)))
    report={'training_seconds':training_seconds,'test_metrics':test.results_dict if test else None,
            'test_status':'NOT_TESTED_deferred' if args.skip_test else 'evaluated',
            'gpu':torch.cuda.get_device_name(0) if device!='cpu' else None,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device!='cpu' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device!='cpu' else None,
            'best_checkpoint':str(destination/'weights/best.pt'),
            'warning':'Pilot dataset only; 3 test images imply extremely high uncertainty. 0407 is a train example.'}
    (destination/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == "__main__":
    main()
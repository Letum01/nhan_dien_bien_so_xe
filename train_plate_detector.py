"""Explicit opt-in YOLO training/validation on human annotated plate boxes."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Human-reviewed YOLO dataset YAML with train/val/test")
    parser.add_argument("architecture", type=Path, help="Local YOLO architecture YAML; no automatic pretrained weight download")
    parser.add_argument("output", type=Path, help="New experiment directory outside saved_models")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    protected = Path(__file__).resolve().parent / "saved_models"
    destination = args.output.resolve()
    if destination.exists() or destination == protected or protected in destination.parents:
        raise ValueError("Use a new experiment directory outside saved_models")
    if not args.data.is_file() or not args.architecture.is_file():
        raise FileNotFoundError("Local data YAML and architecture YAML required")
    from ultralytics import YOLO
    import yaml
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if config.get("names") not in ({0: "license_plate"}, ["license_plate"]) or not all(config.get(s) for s in ("train", "val", "test")):
        raise ValueError("Dataset must define license_plate only and separate train/val/test splits")
    model = YOLO(str(args.architecture))
    model.train(data=str(args.data.resolve()), epochs=args.epochs, pretrained=False,
                project=str(destination.parent), name=destination.name, exist_ok=False)
    trained = YOLO(str(destination / "weights" / "best.pt"))
    trained.val(data=str(args.data.resolve()), split="test", project=str(destination), name="test_metrics")


if __name__ == "__main__":
    main()
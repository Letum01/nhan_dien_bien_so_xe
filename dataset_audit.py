"""Read-only image/annotation inventory; duplicate matches are review candidates."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def fingerprint(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image = cv2.imread(str(path))
    if image is None:
        return digest, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    perceptual = sum(int(bit) << i for i, bit in enumerate(bits))
    return digest, perceptual, image.shape


def audit(root=ROOT):
    root = Path(root)
    rows, hashes, perceptuals = [], defaultdict(list), []
    # Only dataset source roots; debug images/model outputs are not independent samples.
    sources = [p for folder in (root / "dataset", root / "datasets") if folder.exists()
               for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
    annotations = []
    for folder in (root / "dataset", root / "datasets"):
        if not folder.exists():
            continue
        for path in folder.rglob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    annotations.append({"manifest": str(path.relative_to(root)), **row})
        for path in folder.rglob("reviewed_detection.json"):
            for row in json.loads(path.read_text(encoding="utf-8")):
                annotations.append({"manifest": str(path.relative_to(root)), **row})
    for path in sorted(sources):
        digest, phash, shape = fingerprint(path)
        relative = str(path.relative_to(root))
        row = dict(image_path=relative, extension=path.suffix.lower(), readable=shape is not None,
                   width=shape[1] if shape else None, height=shape[0] if shape else None,
                   sha256=digest, phash=f"{phash:016x}" if phash is not None else None,
                   category="UNKNOWN", review_status="unlabeled")
        rows.append(row); hashes[digest].append(relative)
        if phash is not None:
            perceptuals.append((relative, phash))
    # All-pairs Hamming distance for ~3k images (~4.3m inexpensive bit counts).
    near = []
    for i, (path, value) in enumerate(perceptuals):
        for other, candidate in perceptuals[:i]:
            distance = (value ^ candidate).bit_count()
            if distance <= 6:
                near.append({"a": other, "b": path, "hamming": distance,
                             "status": "needs_review_not_verified_duplicate"})
    status_counts = Counter()
    for annotation in annotations:
        # Existing manifests may use explicit human_reviewed boolean; never infer from filename.
        status = annotation.get("review_status", annotation.get("annotation_status", "unlabeled"))
        if annotation.get("human_reviewed") is True:
            status = "reviewed"
        status_counts[status] += 1
    valid = sum(r["readable"] for r in rows)
    return {"source_roots": ["dataset", "datasets"], "total_images": len(rows), "readable_images": valid,
            "corrupt_images": [r["image_path"] for r in rows if not r["readable"]],
            "formats": dict(Counter(r["extension"] for r in rows)),
            "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in rows if r["readable"])),
            "exact_duplicate_groups": [paths for paths in hashes.values() if len(paths) > 1],
            "unique_sha256_count": len(hashes),
            "near_duplicate_candidates": near, "annotation_rows_by_status": dict(status_counts),
            "annotation_rows": annotations, "images": rows,
            "category_distribution": {"UNKNOWN": len(rows)},
            "training_eligible_images": 0,
            "training_eligibility_note": "No eligibility inferred from raw manifest counts; reviewed, validated split export required.",
            "needs_labeling_or_validation": len(rows),
            "missing_groups": ["verified vehicle type", "verified row count", "verified lighting/skew/blur", "verified negative images"],
            "rules": ["Never delete duplicates", "Near-hash matches need human review", "Group duplicates before splitting", "Unreviewed estimates/proposals excluded"]}


def main():
    report = audit()
    folder = ROOT / "reports"
    folder.mkdir(exist_ok=True)
    (folder / "dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (folder / "dataset_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(report["images"][0]) if report["images"] else ["image_path"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(report["images"])
    text = (f"# Dataset audit\n\nImages: {report['total_images']}\n\nReadable: {report['readable_images']}\n\n"
            f"Corrupt: {len(report['corrupt_images'])}\n\nExact duplicate groups: {len(report['exact_duplicate_groups'])}\n\n"
            f"Near duplicate candidate pairs (pHash <=6): {len(report['near_duplicate_candidates'])}\n\n"
            f"Annotation rows: {report['annotation_rows_by_status']}\n\nResolutions: {report['resolutions']}\n\n"
            "Training eligibility: BLOCKED pending reviewed annotations and validated export. Categories UNKNOWN until human metadata exists.\n\n"
            "No images deleted. pHash matches are candidates, not verified duplicate labels. Debug outputs excluded from source counts.\n")
    (folder / "dataset_audit.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
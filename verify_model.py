"""Reproduserbar evaluering av NIH Chest X-Ray-modellen.

Kjør fra prosjektmappen:
    python verify_model.py

Standardverdiene bruker NIH-datasettet i D:/relevant_projects/chest_xray
og best_model.pt i denne mappen. Programmet evaluerer utelukkende den
offisielle holdt-utenfor-testlisten fra NIH og lagrer rapporter i audit_results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_DATA_DIR = Path(r"D:\relevant_projects\chest_xray")
DEFAULT_MODEL_PATH = Path(__file__).with_name("best_model.pt")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("audit_results")

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class NIHTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, paths: dict[str, Path]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.paths = paths

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(self.paths[row["Image Index"]]) as image:
            tensor = EVAL_TRANSFORM(image.convert("RGB"))
        labels = torch.tensor(row[CLASSES].to_numpy(dtype=np.float32))
        return tensor, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluer en bryst-røntgenmodell på NIH sitt offisielle testsett."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--smoke-test", type=int, metavar="N",
        help="Kjør bare de første N testbildene for å kontrollere installasjonen.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nSTOPPET\n{message}", file=sys.stderr)
    raise SystemExit(2)


def validate_inputs(data_dir: Path, model_path: Path) -> tuple[Path, Path, Path]:
    metadata = data_dir / "Data_Entry_2017.csv"
    test_list = data_dir / "test_list.txt"
    if not data_dir.is_dir():
        fail(f"Fant ikke datamappen: {data_dir}")
    if not metadata.is_file() or not test_list.is_file():
        fail("Fant ikke Data_Entry_2017.csv og test_list.txt i datamappen.")
    if not model_path.is_file():
        fail(f"Fant ikke modellfilen: {model_path}")
    return metadata, test_list, model_path


def build_image_index(data_dir: Path) -> dict[str, Path]:
    # NIH leverer hver bildedel med en ekstra undermappe kalt ``images``.
    # Rekursivt søk gjør programmet robust for både den opprinnelige
    # katalogstrukturen og eventuelle senere utpakkingsvarianter.
    images = {path.name: path for path in data_dir.glob("images_*/**/*.png")}
    if len(images) < 100_000:
        fail(f"Fant bare {len(images):,} PNG-filer. Det komplette NIH-datasettet forventes.")
    return images


def build_test_frame(metadata_path: Path, test_list_path: Path, image_paths: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    listed = {line.strip() for line in test_list_path.read_text().splitlines() if line.strip()}
    test = frame[frame["Image Index"].isin(listed)].copy()
    missing = sorted(set(test["Image Index"]) - set(image_paths))
    if missing:
        fail(f"{len(missing):,} testbilder mangler. Første fil som mangler er {missing[0]}.")
    if len(test) != len(listed):
        fail("Testlisten og metadatafilen matcher ikke.")
    for label in CLASSES:
        test[label] = test["Finding Labels"].str.split("|").map(lambda labels: float(label in labels))
    return test


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    # Modellfilen inneholder vekter og enkle NumPy-metadata. Bare disse dataobjektene
    # tillates ved lasting, slik at evalueringsfilen ikke laster vilkårlig kode.
    from numpy._core.multiarray import scalar

    torch.serialization.add_safe_globals([
        np.dtype,
        scalar,
        type(np.dtype(np.float64)),
    ])
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or state.get("fc.weight") is None:
        fail("Modellfilen har ikke et gyldig ResNet50-checkpoint.")
    if tuple(state["fc.weight"].shape) != (len(CLASSES), 2048):
        fail("Modellens klassifiseringslag matcher ikke de 14 NIH-klassene.")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(state)
    return model.to(device).eval()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_prob)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (y_prob >= low) & (y_prob < high if high < 1 else y_prob <= high)
        if selected.any():
            error += selected.mean() * abs(y_true[selected].mean() - y_prob[selected].mean())
    return float(error)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for number, (images, labels) in enumerate(loader, start=1):
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probabilities.append(probabilities)
            all_labels.append(labels.numpy())
            if number % 100 == 0 or number == len(loader):
                print(f"  {number}/{len(loader)} batcher ferdig")
    return np.vstack(all_labels), np.vstack(all_probabilities)


def build_metrics(labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    predictions = probabilities >= 0.5
    for index, name in enumerate(CLASSES):
        y_true = labels[:, index].astype(int)
        y_prob = probabilities[:, index]
        has_both_classes = y_true.min() != y_true.max()
        rows.append({
            "disease": name,
            "positive_cases": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auroc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else np.nan,
            "average_precision": float(average_precision_score(y_true, y_prob)) if has_both_classes else np.nan,
            "brier_score": float(np.mean((y_prob - y_true) ** 2)),
            "ece_15_bins": expected_calibration_error(y_true, y_prob),
            "sensitivity_at_0_5": float(recall_score(y_true, predictions[:, index], zero_division=0)),
            "precision_at_0_5": float(precision_score(y_true, predictions[:, index], zero_division=0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    metadata_path, test_list_path, model_path = validate_inputs(args.data_dir, args.model)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Bruker {device.type.upper()}.")
    print("Bygger bildeindeks.")
    image_paths = build_image_index(args.data_dir)
    test = build_test_frame(metadata_path, test_list_path, image_paths)
    if args.smoke_test:
        test = test.head(args.smoke_test).copy()
        print(f"Smoketest med {len(test)} bilder.")
    else:
        print(f"Full evaluering med {len(test):,} holdt-utenfor-testbilder.")

    loader = DataLoader(
        NIHTestDataset(test, image_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(model_path, device)
    started = time.perf_counter()
    labels, probabilities = evaluate(model, loader, device)
    elapsed = time.perf_counter() - started
    metrics = build_metrics(labels, probabilities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_disease_metrics.csv", index=False)
    np.savez_compressed(args.output_dir / "test_predictions.npz", labels=labels, probabilities=probabilities)
    summary = {
        "model": str(model_path),
        "data_dir": str(args.data_dir),
        "test_images": int(len(test)),
        "device": device.type,
        "elapsed_seconds": round(elapsed, 1),
        "mean_auroc": round(float(metrics["auroc"].mean()), 4),
        "mean_average_precision": round(float(metrics["average_precision"].mean()), 4),
        "mean_brier_score": round(float(metrics["brier_score"].mean()), 4),
        "mean_ece_15_bins": round(float(metrics["ece_15_bins"].mean()), 4),
        "threshold_note": "Sensitivity and precision use a fixed 0.5 threshold. Do not use them as clinical thresholds.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFERDIG")
    print(f"Gjennomsnittlig AUROC: {summary['mean_auroc']}")
    print(f"Resultater skrevet til: {args.output_dir}")


if __name__ == "__main__":
    main()

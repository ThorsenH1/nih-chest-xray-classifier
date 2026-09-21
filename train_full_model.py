"""Tren og evaluer NIH Chest X-ray-modellen på en etterprøvbar måte.

Testlisten fra NIH brukes aldri i treningen. Den er den uavhengige sluttesten.
Kjør via TREN_FULL_MODELL.bat fra denne mappen.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
ROOT = Path(__file__).parent
DEFAULT_DATA = Path("D:/relevant_projects/chest_xray")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ChestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_paths: dict[str, Path], transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform
        self.labels = self.frame[CLASSES].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.frame.iloc[idx]["Image Index"]
        with Image.open(self.image_paths[name]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.labels[idx])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full NIH-trening med separat sluttest.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "full_training_run")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "best_model.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_index(data_dir: Path) -> dict[str, Path]:
    paths = {item.name: item for item in data_dir.glob("images_*/**/*.png")}
    if len(paths) != 112120:
        raise RuntimeError(f"Forventet 112120 PNG-filer, fant {len(paths)}.")
    return paths


def labelled_frame(data_dir: Path, paths: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / "Data_Entry_2017.csv")
    for disease in CLASSES:
        frame[disease] = frame["Finding Labels"].str.contains(disease, regex=False).astype(np.float32)
    frame = frame[frame["Image Index"].isin(paths)].copy()
    if len(frame) != 112120:
        raise RuntimeError("Metadata og bildeindeks stemmer ikke overens.")
    return frame


def make_splits(frame: pd.DataFrame, data_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_names = set((data_dir / "train_val_list.txt").read_text().splitlines())
    test_names = set((data_dir / "test_list.txt").read_text().splitlines())
    if len(train_val_names) != 86524 or len(test_names) != 25596 or train_val_names & test_names:
        raise RuntimeError("NIH-listene er ikke komplette eller overlapper.")
    train_val = frame[frame["Image Index"].isin(train_val_names)].copy()
    test = frame[frame["Image Index"].isin(test_names)].copy()
    patients = train_val["Patient ID"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    validation_patients = set(patients[: round(len(patients) * 0.10)])
    validation = train_val[train_val["Patient ID"].isin(validation_patients)].copy()
    train = train_val[~train_val["Patient ID"].isin(validation_patients)].copy()
    if set(train["Patient ID"]) & set(validation["Patient ID"]):
        raise RuntimeError("Pasientoverlapp mellom trening og validering.")
    if set(train["Patient ID"]) & set(test["Patient ID"]):
        raise RuntimeError("Pasientoverlapp mellom trening og sluttest.")
    return train, validation, test


def load_initial_model(path: Path, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if path.exists():
        torch.serialization.add_safe_globals([np.dtype, np._core.multiarray.scalar, type(np.dtype(np.float64))])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Fortsetter fra {path.name}, epoke {checkpoint.get('epoch', 'ukjent')}.")
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        print("Starter fra ImageNet-vekter.")
    return model.to(device)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def class_metrics(probabilities: np.ndarray, labels: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, disease in enumerate(CLASSES):
        truth, scores = labels[:, index], probabilities[:, index]
        rows.append({
            "disease": disease,
            "positive_cases": int(truth.sum()),
            "auroc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
        })
    return rows


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_rows: list[dict[str, float | str]], history: list[dict[str, float]]) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "disease_labels": CLASSES,
        "val_aucs": [row["auroc"] for row in val_rows],
        "val_auc": float(np.mean([row["auroc"] for row in val_rows])),
        "history": history,
        "image_size": 224,
        "normalization_mean": MEAN,
        "normalization_std": STD,
    }, path)


def main() -> None:
    args = arguments()
    if args.epochs < 1:
        raise ValueError("Antall epoker må være minst 1.")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Bruker {device.type.upper()}.")
    print("Indekserer hele NIH-datasettet.")
    paths = image_index(args.data_dir)
    frame = labelled_frame(args.data_dir, paths)
    train, validation, test = make_splits(frame, args.data_dir, args.seed)
    print(f"Trening: {len(train):,}. Validering: {len(validation):,}. Sluttest: {len(test):,}.")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    loader_args = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers:
        loader_args.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(ChestDataset(train, paths, train_transform), batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(ChestDataset(validation, paths, eval_transform), batch_size=args.batch_size * 2, shuffle=False, **loader_args)
    test_loader = DataLoader(ChestDataset(test, paths, eval_transform), batch_size=args.batch_size * 2, shuffle=False, **loader_args)

    model = load_initial_model(args.init_checkpoint, device)
    pos = train[CLASSES].sum().to_numpy(dtype=np.float32)
    weights = np.minimum((len(train) - pos) / pos, args.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_auc = -float("inf")
    best_path = args.run_dir / "best_model.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
            seen += len(images)
            if batch % 100 == 0:
                print(f"  epoke {epoch}, batch {batch}/{len(train_loader)}, tap {total_loss / seen:.4f}")
        probabilities, labels = evaluate(model, val_loader, device)
        rows = class_metrics(probabilities, labels)
        val_auc = float(np.mean([row["auroc"] for row in rows]))
        entry = {"epoch": float(epoch), "train_loss": total_loss / seen, "val_mean_auroc": val_auc}
        history.append(entry)
        scheduler.step()
        print(f"Epoke {epoch}/{args.epochs}: tap={entry['train_loss']:.4f}, validerings-AUROC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(best_path, model, epoch, rows, history)
            print("  Ny beste modell lagret.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities, labels = evaluate(model, test_loader, device)
    rows = class_metrics(probabilities, labels)
    test_auc = float(np.mean([row["auroc"] for row in rows]))
    with (args.run_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "training_images": int(len(train)),
        "validation_images": int(len(validation)),
        "held_out_test_images": int(len(test)),
        "epochs": args.epochs,
        "best_validation_auroc": best_auc,
        "held_out_test_mean_auroc": test_auc,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 1),
        "note": "For research use only. This model is not a diagnostic device.",
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.run_dir / "training_history.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Lag etterprøvbare, lagrede modellresultater til porteføljen.

Velger fem bilder fra NIH sin offisielle holdt-utenfor-testliste, kjører hvert
bilde gjennom det lokale kontrollpunktet og lagrer originalbilde, Grad-CAM og
modellskårer. Skriptet bruker CPU for ikke å forstyrre en aktiv treningsrunde.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).parent
DATA = Path(r"D:\relevant_projects\chest_xray")
OUTPUT = ROOT.parent / "xray-examples"
MODEL_PATH = ROOT / "best_model.pt"
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
SAMPLES = [
    ("cardiomegaly", "Cardiomegaly"),
    ("effusion", "Effusion"),
    ("pneumothorax", "Pneumothorax"),
    ("hernia", "Hernia"),
    ("emphysema", "Emphysema"),
]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model() -> torch.nn.Module:
    from numpy._core.multiarray import scalar

    torch.serialization.add_safe_globals([np.dtype, scalar, type(np.dtype(np.float64))])
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def index_images() -> dict[str, Path]:
    return {path.name: path for path in DATA.glob("images_*/**/*.png")}


def grad_cam(model: torch.nn.Module, image: torch.Tensor, class_index: int) -> np.ndarray:
    values: dict[str, torch.Tensor] = {}
    target = model.layer4[-1]
    def save_activations(_, __, output) -> None:
        values["activations"] = output

    def save_gradients(_, __, output) -> None:
        values["gradients"] = output[0]

    forward = target.register_forward_hook(save_activations)
    backward = target.register_full_backward_hook(save_gradients)
    try:
        logits = model(image)
        model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()
        weights = values["gradients"].mean(dim=(2, 3), keepdim=True)
        heatmap = functional.relu((weights * values["activations"]).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(heatmap, size=(224, 224), mode="bilinear", align_corners=False)
        array = heatmap[0, 0].detach().numpy()
        return (array - array.min()) / (array.max() - array.min() + 1e-8)
    finally:
        forward.remove()
        backward.remove()


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    red = np.clip(1.8 * heatmap, 0, 1)
    green = np.clip(1.8 * (1 - np.abs(heatmap - 0.5) * 2), 0, 1)
    blue = np.clip(1.8 * (1 - heatmap), 0, 1)
    colors = np.stack([red, green, blue], axis=-1) * 255
    return Image.fromarray(np.uint8(base * 0.55 + colors * 0.45))


def main() -> None:
    if not DATA.is_dir() or not MODEL_PATH.is_file():
        raise SystemExit("Mangler NIH-data eller best_model.pt.")
    OUTPUT.mkdir(exist_ok=True)
    frame = pd.read_csv(DATA / "Data_Entry_2017.csv")
    test_names = set((DATA / "test_list.txt").read_text().splitlines())
    test = frame[frame["Image Index"].isin(test_names)].copy()
    paths = index_images()
    model = load_model()
    output: dict[str, object] = {"model": "best_model.pt", "device": "cpu", "examples": {}}

    for key, label in SAMPLES:
        # Rene enkelteksempler gjør både fasit og Grad-CAM-målet etterprøvbart.
        row = test[test["Finding Labels"].eq(label)].sort_values("Image Index").iloc[0]
        path = paths[row["Image Index"]]
        with Image.open(path) as source:
            original = source.convert("RGB")
        tensor = TRANSFORM(original).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(tensor))[0].numpy()
        target_index = CLASSES.index(label)
        heatmap = grad_cam(model, tensor, target_index)
        original.resize((224, 224)).save(OUTPUT / f"{key}.png")
        overlay(original, heatmap).save(OUTPUT / f"{key}-gradcam.png")
        ranking = np.argsort(probabilities)[::-1]
        output["examples"][key] = {
            "image_id": row["Image Index"],
            "test_label": label,
            "target_score": round(float(probabilities[target_index]), 4),
            "top_scores": [
                {"label": CLASSES[index], "score": round(float(probabilities[index]), 4)}
                for index in ranking[:5]
            ],
        }
        print(f"Laget {key}: {row['Image Index']}")

    (OUTPUT / "predictions.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Ferdig. Resultater er lagret i {OUTPUT}")


if __name__ == "__main__":
    main()

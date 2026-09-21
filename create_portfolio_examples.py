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
from PIL import Image, ImageDraw
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
# Disse klassene har NIH sine faktiske boksannotasjoner, slik at porteføljen kan
# vise testsettets fasit uten å late som Grad-CAM er en medisinsk avgrensning.
SAMPLES = [
    ("cardiomegaly", "Cardiomegaly", "Cardiomegaly"),
    ("effusion", "Effusion", "Effusion"),
    ("pneumothorax", "Pneumothorax", "Pneumothorax"),
    ("mass", "Mass", "Mass"),
    ("nodule", "Nodule", "Nodule"),
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


def draw_ground_truth(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Tegn NIH sin avgrensning og pil på originalbildet, uten modellinformasjon."""
    resized = image.convert("RGB").resize((224, 224))
    scale_x = 224 / image.width
    scale_y = 224 / image.height
    x, y, width, height = box
    left, top = x * scale_x, y * scale_y
    right, bottom = (x + width) * scale_x, (y + height) * scale_y
    draw = ImageDraw.Draw(resized)
    color = (245, 92, 58)
    draw.rectangle((left, top, right, bottom), outline=color, width=3)
    target_x = (left + right) / 2
    target_y = (top + bottom) / 2
    start_x = max(16, min(208, target_x - 40))
    start_y = max(15, target_y - 34)
    draw.line((start_x, start_y, target_x, target_y), fill=color, width=3)
    draw.polygon([(target_x, target_y), (target_x - 8, target_y - 4), (target_x - 3, target_y - 9)], fill=color)
    draw.rectangle((5, 5, 62, 23), fill=(26, 29, 28))
    draw.text((10, 8), "FASIT", fill=(255, 255, 255))
    return resized


def main() -> None:
    if not DATA.is_dir() or not MODEL_PATH.is_file():
        raise SystemExit("Mangler NIH-data eller best_model.pt.")
    OUTPUT.mkdir(exist_ok=True)
    frame = pd.read_csv(DATA / "Data_Entry_2017.csv")
    test_names = set((DATA / "test_list.txt").read_text().splitlines())
    boxes = pd.read_csv(DATA / "BBox_List_2017.csv")
    paths = index_images()
    model = load_model()
    output: dict[str, object] = {"model": "best_model.pt", "device": "cpu", "examples": {}}

    for key, label, annotation_label in SAMPLES:
        # Velg blant holdt-utenfor bilder med NIH-boks og ta det sterkeste
        # modelltreffet fra et begrenset, deterministisk utvalg. Dette er en
        # demonstrasjon av arbeidsflyten, mens totalresultatet står separat.
        candidates = boxes[
            (boxes["Finding Label"].eq(annotation_label))
            & (boxes["Image Index"].isin(test_names))
        ].sort_values("Image Index").head(25)
        if candidates.empty:
            raise RuntimeError(f"Fant ingen testannotasjoner for {annotation_label}.")
        target_index = CLASSES.index(label)
        best: tuple[float, pd.Series, Image.Image, np.ndarray] | None = None
        for _, candidate in candidates.iterrows():
            with Image.open(paths[candidate["Image Index"]]) as source:
                candidate_image = source.convert("RGB")
            candidate_tensor = TRANSFORM(candidate_image).unsqueeze(0)
            with torch.inference_mode():
                candidate_scores = torch.sigmoid(model(candidate_tensor))[0].numpy()
            score = float(candidate_scores[target_index])
            if best is None or score > best[0]:
                best = (score, candidate, candidate_image, candidate_scores)
        assert best is not None
        _, row, original, probabilities = best
        tensor = TRANSFORM(original).unsqueeze(0)
        heatmap = grad_cam(model, tensor, target_index)
        original.resize((224, 224)).save(OUTPUT / f"{key}.png")
        box = tuple(float(row[column]) for column in ("Bbox [x", "y", "w", "h]"))
        draw_ground_truth(original, box).save(OUTPUT / f"{key}-fasit.png")
        overlay(original, heatmap).save(OUTPUT / f"{key}-gradcam.png")
        ranking = np.argsort(probabilities)[::-1]
        output["examples"][key] = {
            "image_id": row["Image Index"],
            "test_label": label,
            "bbox": [round(value, 2) for value in box],
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

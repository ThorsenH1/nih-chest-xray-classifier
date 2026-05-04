"""
NIH Chest X-Ray Classifier — Local Inference App
=================================================
Laster inn den trente ResNet50-modellen og kjører en Gradio-demo lokalt.

Bruk:
    python app.py

Åpner automatisk http://localhost:7860 i nettleseren.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import models, transforms
import gradio as gr
import io
import argparse
import os
import sys

# ── Konfigurasjon ─────────────────────────────────────────────────────────────

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

DISEASE_INFO = {
    "Atelectasis"       : {"no": "Atelektase",        "what": "Partial or complete collapse of lung tissue.", "looks": "Increased density in part of the lung, often with structural displacement.", "where": "Most common in lower lobes, particularly posterior and right-sided."},
    "Cardiomegaly"      : {"no": "Kardiomegali",      "what": "Enlarged heart due to heart failure, hypertension, or valve disease.", "looks": "Heart width exceeds 50% of the inner chest width (cardiothoracic ratio > 0.5).", "where": "Central chest — the heart occupies more space than normal."},
    "Consolidation"     : {"no": "Konsolidering",     "what": "Air replaced by fluid, pus, or blood. Common in bacterial pneumonia.", "looks": "Dense white area. May show 'air bronchogram' (dark branching lines through white area).", "where": "Can affect a segment, a lobe, or an entire lung."},
    "Edema"             : {"no": "Lungeødem",          "what": "Fluid leaking into lung tissue. Usually a sign of heart failure.", "looks": "Uniform haziness — 'butterfly pattern' around the lung root.", "where": "Symmetric, starts centrally and spreads outward."},
    "Effusion"          : {"no": "Pleural effusjon",   "what": "Fluid accumulation in the pleural space.", "looks": "Gray/white opacity in lower chest with smooth upper border that follows gravity.", "where": "Lower chest, most often along the sides."},
    "Emphysema"         : {"no": "Emfysem",            "what": "Destruction of air sacs — lungs become hyperinflated. Most common cause: smoking.", "looks": "Dark, overinflated lungs. Flattened diaphragm. Sparse vascular markings.", "where": "Upper lobes affected first."},
    "Fibrosis"          : {"no": "Lungefibrose",       "what": "Scar tissue replacing normal lung parenchyma.", "looks": "Reticular (net-like) irregular lines. Reduced lung volume.", "where": "Typically lower and peripheral lung zones."},
    "Hernia"            : {"no": "Hernie",             "what": "Abdominal organs herniated through the diaphragm into the chest.", "looks": "Unusual structures in the chest — may resemble bowel loops above the diaphragm.", "where": "Usually left-sided or midline."},
    "Infiltration"      : {"no": "Infiltrat",          "what": "Non-specific: inflammation, fluid, or blood partially fills lung tissue.", "looks": "Hazy, cloud-like opacities. Less dense than consolidation.", "where": "Variable — can affect any part of the lung."},
    "Mass"              : {"no": "Masse",               "what": "A discrete opacity > 3 cm. Always requires malignancy workup.", "looks": "Well-defined, round or lobulated area. Spiculated edges suggest malignancy.", "where": "Can occur anywhere in the lung."},
    "Nodule"            : {"no": "Nodule",              "what": "A discrete opacity ≤ 3 cm. May be benign (granuloma) or malignant.", "looks": "Small, round, well-defined white dot.", "where": "Can occur anywhere."},
    "Pleural_Thickening": {"no": "Pleural fortykning", "what": "Thickened pleural membranes — often post-inflammatory.", "looks": "White line along the chest wall, smoother than effusion.", "where": "Along the edge of the lung against the chest wall."},
    "Pneumonia"         : {"no": "Lungebetennelse",    "what": "Lung infection — bacterial, viral, or fungal.", "looks": "Consolidation or infiltrates — white opacities, often with air bronchogram.", "where": "Lower lobes most commonly affected."},
    "Pneumothorax"      : {"no": "Pneumothorax",       "what": "Air in the pleural space causing lung collapse.", "looks": "Sharp line (visceral pleura) parallel to the chest wall. No lung markings beyond this line.", "where": "Apex of the lung."},
}

CONFIDENCE_LEVELS = [
    (0.70, "🔴 Strong",   "The model is highly confident."),
    (0.50, "🟠 Moderate", "Clear signs present — warrants evaluation."),
    (0.35, "🟡 Weak",     "Possible subtle finding. Uncertain."),
    (0.00, "⚪ None",      "No significant signs detected."),
]

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Modell-lasting ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Last inn ResNet50-modellen fra checkpoint-fil."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Fant ikke modell-filen: {checkpoint_path}\n"
            f"Last ned best_model.pt fra Google Drive og oppgi stien med --model"
        )

    print(f"Laster modell fra: {checkpoint_path}")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(DISEASE_LABELS))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    val_auc = ckpt.get("val_auc", "?")
    epoch   = ckpt.get("epoch", "?")
    print(f"✅ Modell lastet  (epoke={epoch}, val_AUC={val_auc:.4f})")
    return model


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Viser hvilke piksler i bildet som påvirket prediksjonen mest.
    Kjøres på det siste konvlusjonslaget i ResNet50 (layer4[-1]).
    Rødt = høy aktivasjon, blått = lav aktivasjon.
    """
    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        # Hekt på det siste residual-blokket
        model.layer4[-1].register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        model.layer4[-1].register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        # Vekt aktivasjonskart med gjennomsnittsgradient (global average pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img_np      = np.array(pil_img.convert("RGB"))
    heatmap_rgb = (cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended     = (img_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)


# ── Transforms ────────────────────────────────────────────────────────────────

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Prediksjon og forklaring ──────────────────────────────────────────────────

def get_confidence(prob: float):
    for threshold, label, desc in CONFIDENCE_LEVELS:
        if prob >= threshold:
            return label, desc
    return CONFIDENCE_LEVELS[-1][1], CONFIDENCE_LEVELS[-1][2]


def predict_and_explain(image: Image.Image, model: torch.nn.Module,
                        gradcam: GradCAM, device: torch.device):
    """Kjør prediksjon + GradCAM på ett bilde. Returnerer (figur, markdown-tekst)."""
    if image is None:
        return None, "Last opp et røntgenbilde for å starte."

    img_resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    tensor      = eval_transform(img_resized).unsqueeze(0).to(device)

    # Inferens
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).float().cpu().numpy()[0]

    sorted_idx   = np.argsort(probs)[::-1]
    positive_idx = [i for i in sorted_idx if probs[i] >= 0.35]

    # GradCAM for topp-funn (krever gradienter)
    heatmap_img = None
    if len(positive_idx) > 0:
        t   = eval_transform(img_resized).unsqueeze(0).to(device)
        cam = gradcam.generate(t, positive_idx[0])
        heatmap_img = overlay_heatmap(img_resized, cam)

    # ── Figur ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")

    ax_orig = fig.add_axes([0.02, 0.35, 0.22, 0.55])
    ax_heat = fig.add_axes([0.02, 0.05, 0.22, 0.28])

    ax_orig.imshow(img_resized, cmap="gray")
    ax_orig.set_title("Input Image", color="white", fontsize=11, pad=6)
    ax_orig.axis("off")

    if heatmap_img:
        top_name = DISEASE_INFO[DISEASE_LABELS[positive_idx[0]]]["no"]
        ax_heat.imshow(heatmap_img)
        ax_heat.set_title(f"GradCAM: {top_name}\n(red = high activation)",
                          color="#ff9f43", fontsize=9, pad=4)
    else:
        ax_heat.text(0.5, 0.5, "No findings\ndetected",
                     ha="center", va="center", color="gray", fontsize=12)
        ax_heat.set_facecolor("#0f0f23")
    ax_heat.axis("off")

    # Sannsynlighetsbars
    ax_bar = fig.add_axes([0.28, 0.42, 0.38, 0.52])
    bar_colors = []
    for i in sorted_idx[:8]:
        p = probs[i]
        if   p >= 0.70: bar_colors.append("#e74c3c")
        elif p >= 0.50: bar_colors.append("#e67e22")
        elif p >= 0.35: bar_colors.append("#f1c40f")
        else:           bar_colors.append("#2c3e50")

    bar_labels = [DISEASE_INFO[DISEASE_LABELS[i]]["no"] for i in sorted_idx[:8]]
    bar_vals   = [probs[i] for i in sorted_idx[:8]]
    bars = ax_bar.barh(bar_labels[::-1], bar_vals[::-1], color=bar_colors[::-1], height=0.6)
    ax_bar.set_xlim(0, 1)
    ax_bar.axvline(0.50, color="white",  lw=1, ls="--", alpha=0.4, label="50%")
    ax_bar.axvline(0.35, color="yellow", lw=1, ls=":",  alpha=0.4, label="35%")
    for bar, val in zip(bars, bar_vals[::-1]):
        ax_bar.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", color="white", fontsize=10, fontweight="bold")
    ax_bar.set_facecolor("#0f0f23")
    ax_bar.tick_params(colors="white", labelsize=10)
    ax_bar.spines[:].set_color("#333355")
    ax_bar.set_title("Predicted Probability (top 8)", color="white", fontsize=11, pad=8)
    ax_bar.legend(loc="lower right", fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    # Forklaringspanel
    ax_text = fig.add_axes([0.68, 0.02, 0.30, 0.94])
    ax_text.set_facecolor("#0f0f23")
    ax_text.axis("off")

    if len(positive_idx) == 0:
        title_txt = "✅ NO PATHOLOGY DETECTED"
        title_col = "#2ecc71"
        body_txt  = (
            "The model finds no significant signs of\n"
            "the 14 conditions it was trained to detect.\n\n"
            "This may indicate:\n"
            " • A normal-appearing chest radiograph\n"
            " • Findings outside the model's scope\n"
            "   (e.g. fractures, tuberculosis)\n"
            " • Subtle findings below detection threshold"
        )
    else:
        top_d      = DISEASE_LABELS[positive_idx[0]]
        info       = DISEASE_INFO[top_d]
        c_lbl, c_desc = get_confidence(probs[positive_idx[0]])
        title_txt  = f"{c_lbl}\n{info['no'].upper()}"
        title_col  = "#e74c3c" if probs[positive_idx[0]] >= 0.50 else "#f1c40f"
        body_txt   = (
            f"WHAT IS IT?\n{info['what']}\n\n"
            f"HOW DOES IT APPEAR?\n{info['looks']}\n\n"
            f"WHERE IN THE IMAGE?\n{info['where']}\n\n"
            f"MODEL CONFIDENCE:\n{c_desc}"
        )
        if len(positive_idx) > 1:
            body_txt += "\n\nOTHER FINDINGS ABOVE 35%:"
            for i in positive_idx[1:4]:
                d = DISEASE_LABELS[i]
                body_txt += f"\n • {DISEASE_INFO[d]['no']}: {probs[i]:.0%}"

    ax_text.text(0.05, 0.97, title_txt, transform=ax_text.transAxes,
                 color=title_col, fontsize=12, fontweight="bold", va="top")
    ax_text.text(0.05, 0.75, body_txt, transform=ax_text.transAxes,
                 color="#ecf0f1", fontsize=9.5, va="top", linespacing=1.6)
    ax_text.text(0.05, 0.04,
                 "⚠️  Educational purposes only.\nNot for clinical use.",
                 transform=ax_text.transAxes, color="#7f8c8d", fontsize=8, va="bottom")

    fig.text(0.50, 0.97, "🫁  NIH Chest X-Ray Analysis",
             ha="center", color="white", fontsize=14, fontweight="bold")
    fig.text(0.50, 0.93,
             "ResNet50 | Transfer Learning | 14 thoracic disease classes | 112,120 training images",
             ha="center", color="#7f8c8d", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    result_img = Image.open(buf).copy()
    buf.close()

    # Markdown-oppsummering
    if len(positive_idx) == 0:
        md_out = "### ✅ No pathology detected\n\nThe model finds none of the 14 diseases in this image."
    else:
        lines = ["### 🔍 Findings\n"]
        for i in positive_idx[:5]:
            d = DISEASE_LABELS[i]
            c_lbl, _ = get_confidence(probs[i])
            lines.append(f"**{DISEASE_INFO[d]['no']}** — {probs[i]:.0%}  {c_lbl}")
            lines.append(f"> {DISEASE_INFO[d]['what']}\n")
        lines.append("---\n*⚠️ For educational purposes only — not for clinical diagnosis.*")
        md_out = "\n".join(lines)

    return result_img, md_out


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(model: torch.nn.Module, gradcam: GradCAM, device: torch.device) -> gr.Blocks:
    def predict_wrapper(image):
        return predict_and_explain(image, model, gradcam, device)

    with gr.Blocks(theme=gr.themes.Base(), title="Chest X-Ray Analysis") as demo:
        gr.Markdown("""
        # 🫁 NIH Chest X-Ray Analysis
        **ResNet50 fine-tuned on 112,120 chest radiographs | 14 thoracic disease classes**

        Upload a frontal chest X-ray to receive:
        - 📊 **Predicted probability** for all 14 diseases
        - 🗺️ **GradCAM heatmap** — visualizes *where* in the image the model detects pathology
        - 📖 **Clinical explanation** — what each finding means and where to look

        > ⚠️ For educational and demonstration purposes only — not for clinical diagnosis.
        """)

        with gr.Row():
            inp = gr.Image(type="pil", label="Upload chest X-ray", height=320)
            btn = gr.Button("🔍 Analyse", variant="primary", scale=0)

        with gr.Row():
            out_img = gr.Image(label="Analysis with GradCAM", type="pil", height=520)

        out_md = gr.Markdown()

        btn.click(fn=predict_wrapper, inputs=inp, outputs=[out_img, out_md])
        inp.change(fn=predict_wrapper, inputs=inp, outputs=[out_img, out_md])

    return demo


# ── Hovedprogram ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIH Chest X-Ray Classifier — Local Demo")
    parser.add_argument(
        "--model", type=str, default="best_model.pt",
        help="Sti til best_model.pt (default: ./best_model.pt)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port å kjøre Gradio på (default: 7860)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Tving CPU-modus selv om GPU er tilgjengelig"
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Kjører på: {device}")

    model   = load_model(args.model, device)
    gradcam = GradCAM(model)
    demo    = build_ui(model, gradcam, device)

    print(f"\n🚀 Åpner demo på http://localhost:{args.port}")
    demo.launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()

# 🫁 NIH Chest X-Ray Classifier

End-to-end deep learning-pipeline for multi-label klassifisering av 14 thorax-sykdommer
fra røntgenbilder, bygget med PyTorch og transfer learning.

**Dataset:** [NIH Chest X-Ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) — 112 120 bilder, 30 805 pasienter  
**Modell:** ResNet50 pretrained på ImageNet  
**Test mean AUC: 0.795**

## 📊 Resultater

| Sykdom | AUC |
|---|---|
| Hernia | 0.928 |
| Emphysema | 0.895 |
| Cardiomegaly | 0.875 |
| Pneumothorax | 0.858 |
| Edema | 0.833 |
| Effusion | 0.809 |
| Fibrosis | 0.807 |
| Mass | 0.770 |
| Pleural_Thickening | 0.760 |
| Atelectasis | 0.748 |
| Consolidation | 0.731 |
| Nodule | 0.715 |
| Pneumonia | 0.703 |
| Infiltration | 0.699 |

![ROC-kurver](results/roc_curves.png)
![Treningshistorikk](results/training_history.png)

## 🏗️ Arkitektur

- **Backbone:** ResNet50 pretrained på ImageNet (transfer learning)
- **Output:** 14 sigmoid-outputs (multi-label — ett bilde kan ha flere sykdommer)
- **Loss:** BCEWithLogitsLoss med pos\_weight for klasseubalanse
- **Optimizer:** AdamW + CosineAnnealingLR
- **Precision:** Mixed precision FP16

## 🧠 Viktige designvalg

**Multi-label, ikke multi-class:** Et røntgenbilde kan vise flere sykdommer samtidig.
Derfor brukes én sigmoid per klasse og BCE-loss, ikke softmax og cross-entropy.

**Pasient-respektert splitting:** Samme pasient kan ikke være i både trening og test —
forhindrer data leakage, som er den vanligste feilen i medisinsk ML.

**AUC som metrikk:** ~54% av bildene er "No Finding". Accuracy ville vært 90%+ bare
ved å gjette "frisk" på alt. AUC per sykdom er den riktige metrikken.

## 🚀 Kjør selv

```bash
pip install -r requirements.txt
# Åpne notebooks/nih_chest_xray_classifier.ipynb i Google Colab
```

## 🔧 Tech stack
PyTorch · torchvision · scikit-learn · Gradio · Kaggle API · Google Colab T4 GPU

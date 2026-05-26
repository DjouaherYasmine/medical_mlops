---
title: ChestMNIST API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Topic 10 — MLOps Pipeline for Medical Model Deployment

End-to-end MLOps pipeline for training, versioning, deploying, and **monitoring** a medical imaging model on **ChestMNIST** (14-class multi-label chest X-ray classification).

**Team:** F6-Score · ESI Algiers · Advanced Machine Learning 2025–2026  
**Stack:** PyTorch · MedMNIST · MLflow · FastAPI · Streamlit · Python 3.11

---

## Results Summary

| Model            | Strategy       | Test AUC   | F1 tuned | ms/img   |
| ---------------- | -------------- | ---------- | -------- | -------- |
| MobileNetV2      | Linear probe   | 0.6852     | —        | —        |
| ResNet18         | Linear probe   | 0.6865     | —        | —        |
| MobileNetV2      | Full fine-tune | **0.7900** | 0.2360   | **58.6** |
| ResNet18         | Full fine-tune | 0.7820     | 0.2331   | 87.3     |
| Yang et al. SOTA | Full fine-tune | 0.7707     | —        | —        |

Both fine-tuned models exceed SOTA. MobileNetV2 is deployed (higher AUC, 1.5× faster, 5× fewer params).

---

## Project Structure

```
medical_mlops/
├── data/ ← ChestMNIST auto-downloaded here (gitignored)
├── notebooks/
│ ├── W2_data_exploration.ipynb
│ ├── W2_baseline_experiments.ipynb
│ ├── W3_finetuning.ipynb
│ └── W3_evaluation.ipynb
├── src/
│ ├── api.py ← FastAPI serving (3 endpoints)
│ ├── config.py
│ ├── data_loader.py
│ └── mlflow_setup.py
├── dashboard/
│ └── app.py ← Streamlit monitoring dashboard
├── scripts/
│ ├── export_test_images.py ← Exports hospital simulation images
│ └── restore_checkpoints.py ← Restores .pth from MLflow artifacts
├── experiments/
│ └── mlflow.db ← All experiment runs (committed)
├── models/ ← .pth checkpoints (gitignored, restore via script)
├── test_images/ ← Hospital simulation images (gitignored, regenerate)
│ ├── hospital_A/ ← Reference (clean)
│ ├── hospital_B/ ← Brightness degradation (covariate shift)
│ ├── hospital_C/ ← Resolution drop at img 30 (sudden shift)
│ └── hospital_D/ ← Rare class oversampling (label shift)
├── logs/
│ └── inference_log.jsonl ← Live inference log (gitignored)
├── figures/ ← All W2+W3 plots
├── reports/
│ ├── W2_baseline_report.pdf
│ └── W3_experiments_summary.pdf
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup

```bash
# 1. Clone and create environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Restore model checkpoints from MLflow
python scripts/restore_checkpoints.py

# 4. Generate hospital simulation images
python scripts/export_test_images.py
```

---

## Dataset

ChestMNIST — 14-class multi-label chest X-ray classification (MedMNIST benchmark).  
Train: 78,468 · Val: 11,219 · Test: 22,433 · Native resolution: 28×28 → resized to 224×224.

Downloaded automatically on first run:

```python
from medmnist import ChestMNIST
ds = ChestMNIST(split="test", download=True)
```

---

## Running the Full System

### 1 — MLflow experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --port 5000
# Open http://localhost:5000
```

### 2 — FastAPI serving

```bash
uvicorn src.api:app --port 8000
# Open http://localhost:8000/docs
```

Endpoints:

- `GET /health` — model load status
- `POST /predict/{model_name}` — inference with tuned thresholds (`mobilenet` or `resnet`)
- `POST /predict/ab` — randomised A/B routing between both models

### 3 — Monitoring dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

Simulates 4 hospitals with different acquisition profiles, streams images to the API, and runs a **Page-Hinkley Test** (δ=0.005, λ=50) per hospital to detect drift in real time. Drift events are logged to MLflow with `hospital_id` and `drift_type` tags.

### 4 — Docker

```bash
docker build -t chestmnist-api .
docker run -p 8000:8000 chestmnist-api
```

---

## MLOps Practices Implemented

| Practice                         | Implementation                                                           |
| -------------------------------- | ------------------------------------------------------------------------ |
| Experiment tracking              | MLflow — params, per-epoch metrics, per-class AUC, artifacts             |
| Model versioning                 | Checkpoints as MLflow artifacts, restore via script                      |
| Reproducibility                  | Seed 42, pinned requirements.txt, official MedMNIST splits               |
| Serving                          | FastAPI with input validation, tuned thresholds, latency logging         |
| A/B testing                      | `/predict/ab` endpoint — random routing between MobileNetV2 and ResNet18 |
| Training-serving skew prevention | Identical preprocessing pipeline in training and API                     |
| Drift detection                  | Page-Hinkley Test on live confidence stream, per-hospital                |
| Drift logging                    | MLflow runs with `hospital_id`, `drift_type`, `pht_value` tags           |
| Inference logging                | JSONL log at `logs/inference_log.jsonl`                                  |
| Containerisation                 | Dockerfile for API serving                                               |

---

## References

- Yang et al., MedMNIST v2, Scientific Data 2023
- Sandler et al., MobileNetV2, CVPR 2018
- He et al., ResNet, CVPR 2016
- Sculley et al., Hidden Technical Debt in ML Systems, NeurIPS 2015
- Zaharia et al., MLflow, IEEE Data Engineering Bulletin 2018
- Bifet & Gavalda, ADWIN, SDM 2007
- Gama et al., DDM, 200

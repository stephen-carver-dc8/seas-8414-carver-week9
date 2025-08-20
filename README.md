# seas-8414-carver-week9 - Prescriptive DGA Detector

## Goals
- Train a domain classifier with [H2O AutoML](https://docs.h2o.ai/).
- Explain model decisions using [SHAP](https://shap.readthedocs.io/).
- Turn explanations into a prescriptive incident-response playbook via Google Generative AI.

## Architecture
```
1_train_and_export.py
├─ Generates synthetic DGA/legit data
├─ Trains AutoML classifier
└─ Exports leader as `models/DGA_Leader.zip`

2_analyze_domain.py
├─ Computes features (length, entropy) for a domain
├─ Scores with the exported MOJO
├─ Explains the prediction with SHAP
└─ Summarizes the findings and asks Gemini for a playbook
```
`models/DGA_Leader.zip` and the training CSV are included so the app can run immediately.

## Usage
1. **(Optional) Retrain the model**
   ```bash
   python 1_train_and_export.py
   ```
   This regenerates the training CSV and refreshes the MOJO model.
2. **Analyze a domain**
   ```bash
   python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
   ```
   Set `GEMINI_API_KEY` in the environment to receive a live playbook from Gemini. If the key is missing, a deterministic fallback playbook is returned.

## Examples
- Legit domain:
  ```bash
  python 2_analyze_domain.py --domain google.com
  ```
  Outputs a low DGA probability and no playbook.
- DGA-like domain:
  ```bash
  python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
  ```
  Prints a 3–4 step incident-response playbook.

## Requirements
Install Python dependencies with:
```bash
pip install -r requirements.txt
```
Java (JRE) is required for H2O.
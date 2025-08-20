# 2_analyze_domain.py
import argparse
import json
import math
import os
from pathlib import Path

import h2o
import numpy as np
import pandas as pd
import shap
import httpx

# --- Config ---
MODEL_ZIP = Path("models/DGA_Leader.zip")     # Produced by 1_train_and_export.py
TRAIN_CSV = Path("data/dga_dataset_train.csv")    # Used for SHAP background
POS_LABEL = "dga"                            # Positive class
FEATURES = ["length", "entropy"]             # Matches lab
GENAI_MODEL = "gemini-2.5-flash-preview-05-20"
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GENAI_MODEL}:generateContent"

# --- Helpers ---
def get_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    n = float(len(s))
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((cnt / n) * math.log(cnt / n, 2) for cnt in freq.values())

def load_mojo(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"MOJO not found at {path}. Run 1_train_and_export.py first.")
    # import_mojo loads a production-scoring model inside H2O
    return h2o.import_mojo(str(path))

def positive_prob_column(pred_df: pd.DataFrame, positive_label: str) -> str:
    # H2O predict frame: first col 'predict' + one prob col per class (names vary)
    for c in pred_df.columns:
        if c.lower() == positive_label.lower():
            return c
    prob_cols = [c for c in pred_df.columns if c != "predict"]
    if not prob_cols:
        raise ValueError("No probability columns found in predictions.")
    return prob_cols[-1]  # fallback: last prob column

async def generate_playbook_genai(domain, xai_findings: str) -> str:
    if not GENAI_API_KEY:
        # Offline fallback: deterministic 3–4 steps
        lines = [
            f"Domain: {domain}",
            "1. Isolate the source host from the network (EDR containment or VLAN quarantine).",
            "2. Block the domain at DNS, proxy, and firewall; add it to your blocklists.",
            "3. Collect and preserve evidence (DNS logs, EDR timeline, process list, persistence, outbound connections).",
            "4. Remediate or reimage if malware is found; rotate credentials and monitor for re-queries."
        ]
        return "\n".join(lines)

    system_prompt = (
        "As a SOC Manager, write a prescriptive incident-response playbook for a Tier 1 analyst.\n"
        "Rules:\n"
        "1) Do NOT explain the AI model.\n"
        "2) Output ONLY a numbered list with 3–4 concise steps.\n"
        "3) Each step starts with a verb and is 1 sentence.\n"
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\nAlert Details & Context:\n{xai_findings}"}]}
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GENAI_URL,
            params={"key": GENAI_API_KEY},
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload),
        )
        if r.status_code != 200:
            return f"Error from GenAI: {r.status_code} {r.text}"

        data = r.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text:
            return "Error: Unexpected GenAI response."
        # Keep first 3–4 numbered lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        steps = [ln for ln in lines if ln.strip().startswith(tuple(f"{i}." for i in range(1, 10)))]
        return "\n".join(steps[:4]) if steps else text.strip()

def summarize_shap(domain: str, prob_dga: float, instance_df: pd.DataFrame, shap_values: np.ndarray, background_df: pd.DataFrame) -> str:
    """
    Turn single-row SHAP values into a human-readable, structured summary.
    We also tag features as 'high'/'low' vs background using quartiles.
    """
    # Choose the row (only one)
    row = instance_df.iloc[0]
    vals = shap_values[0]  # shape (n_features,)

    # Rank features by absolute impact
    order = np.argsort(-np.abs(vals))
    lines = []
    # Precompute quartiles for "high/low" wording
    q1 = background_df.quantile(0.25)
    q3 = background_df.quantile(0.75)

    for idx in order:
        feat = instance_df.columns[idx]
        value = row.iloc[idx]
        sv = vals[idx]
        direction = "pushed the prediction towards 'dga'" if sv > 0 else "pushed the prediction towards 'legit'"
        # High/low tagging
        tag = "value"
        try:
            if value >= q3[feat]:
                tag = "high value"
            elif value <= q1[feat]:
                tag = "low value"
        except Exception:
            pass
        lines.append(f" - A {tag} of '{feat}' = {round(float(value), 4)} ({direction}).")

    conf_pct = round(100.0 * prob_dga, 1)
    summary = (
        f"- Alert: Potential DGA domain detected.\n"
        f"- Domain: '{domain}'\n"
        f"- AI Model Explanation (from SHAP): The model flagged this domain with {conf_pct}% confidence.\n"
        f"  The classification was primarily driven by:\n" + "\n".join(lines[:3])  # keep top 3 drivers
    )
    return summary

def main():
    parser = argparse.ArgumentParser(description="Analyze a domain with H2O MOJO + SHAP + GenAI.")
    parser.add_argument("--domain", required=True, help="Domain name to analyze (e.g., kq3v9z7j1x5f8g2h.info)")
    args = parser.parse_args()
    domain = args.domain.strip()

    # Feature engineering for single domain
    features_pdf = pd.DataFrame([{
        "length": len(domain),
        "entropy": get_entropy(domain),
    }], columns=FEATURES)

    # Start H2O and load MOJO
    h2o.init()
    mojo_model = load_mojo(MODEL_ZIP)

    # Predict
    h2o_row = h2o.H2OFrame(features_pdf)
    pred_pdf = mojo_model.predict(h2o_row).as_data_frame()
    prob_col = positive_prob_column(pred_pdf, POS_LABEL)
    prob_dga = float(pred_pdf[prob_col].iloc[0])
    predicted = pred_pdf["predict"].iloc[0]

    print(f"Predicted class: {predicted}")
    print(f"Probability {POS_LABEL}: {prob_dga:.4f}")

    # Only explain and prescribe if it looks like DGA
    if str(predicted).lower() == POS_LABEL:
        # Build SHAP background from training CSV
        if not TRAIN_CSV.exists():
            h2o.shutdown(prompt=False)
            raise FileNotFoundError(f"{TRAIN_CSV} not found. Run 1_train_and_export.py first.")

        train_pdf = pd.read_csv(TRAIN_CSV)
        background_pdf = train_pdf[FEATURES].copy()
        # Keep SHAP light but meaningful
        background = shap.kmeans(background_pdf, min(25, len(background_pdf)))

        # Wrap predict->positive probability for SHAP
        def predict_pos(arr_like):
            pdf = pd.DataFrame(arr_like, columns=FEATURES)
            preds = mojo_model.predict(h2o.H2OFrame(pdf)).as_data_frame()
            pcol = positive_prob_column(preds, POS_LABEL)
            return preds[pcol].values

        explainer = shap.KernelExplainer(predict_pos, background)
        shap_vals = explainer.shap_values(features_pdf)

        # SHAP may return list per class; choose positive class
        if isinstance(shap_vals, list):
            shap_arr = shap_vals[-1]
        else:
            shap_arr = shap_vals

        # Build xai_findings text
        xai_text = summarize_shap(domain, prob_dga, features_pdf, shap_arr, background_pdf)

        # GenAI playbook
        playbook = asyncio_run(generate_playbook_genai(domain, xai_text))
        print("\n=== Prescriptive Playbook ===")
        print(playbook)
        with open("prescriptive_playbook.txt", "w") as f:
            f.write("=== Prescriptive Playbook ===\n")
            f.write(str(playbook) + "\n")
    else:
        print("\nNo prescriptive playbook generated (prediction is 'legit').")
        with open("prescriptive_playbook.txt", "w") as f:
            f.write(f"Domain: {domain}\n")
            f.write("No prescriptive playbook generated (prediction is 'legit').\n")

    h2o.shutdown(prompt=False)

# Simple asyncio runner for 3.7+
def asyncio_run(coro):
    try:
        import asyncio
        return asyncio.run(coro)
    except RuntimeError:
        # If an event loop is already running (e.g., in some notebooks)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)

if __name__ == "__main__":
    main()

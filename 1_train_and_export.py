# 1_train_and_export.py
import csv
import math
import random
import os
from pathlib import Path

import h2o
from h2o.automl import H2OAutoML

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = Path("data")
CSV_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CSV = Path(CSV_DIR / "dga_dataset_train.csv")

# ---------- Data generation (matches lab: features = length, entropy) ----------
def get_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    n = float(len(s))
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((cnt / n) * math.log(cnt / n, 2) for cnt in freq.values())

def generate_csv(path: Path, n_legit: int = 200, n_dga: int = 200) -> None:
    header = ["domain", "length", "entropy", "class"]
    legit_domains = ["google", "facebook", "amazon", "github", "wikipedia", "microsoft"]
    rows = []

    # legit
    for _ in range(n_legit):
        sld = random.choice(legit_domains)
        domain = f"{sld}.com"
        rows.append([domain, len(domain), get_entropy(domain), "legit"])

    # dga-like
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(n_dga):
        length = random.randint(15, 25)
        sld = "".join(random.choice(alphabet) for _ in range(length))
        domain = f"{sld}.com"
        rows.append([domain, len(domain), get_entropy(domain), "dga"])

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------- Training + MOJO export ----------
def export_mojo_from_leaderboard(aml, out_zip_path: Path) -> str:
    """
    Try leader first; if it cannot export MOJO, try next models on the leaderboard.
    Returns the actual path written by H2O.
    """
    # Pull model ids top-to-bottom
    lb = aml.leaderboard.as_data_frame()
    model_ids = lb["model_id"].tolist()

    for mid in model_ids:
        try:
            model = h2o.get_model(mid)
            # Newer H2O: model.download_mojo(...)
            if hasattr(model, "download_mojo"):
                mojo_path = model.download_mojo(path=str(out_zip_path.parent), get_genmodel_jar=False)
            else:
                # Fallback, older API
                from h2o import download_mojo  # type: ignore
                mojo_path = download_mojo(model=model, path=str(out_zip_path.parent), get_genmodel_jar=False)

            # Rename to desired filename if different
            if Path(mojo_path).resolve() != out_zip_path.resolve():
                os.replace(mojo_path, out_zip_path)
            print(f"Exported MOJO from model '{mid}' to: {out_zip_path}")
            return str(out_zip_path)
        except Exception as e:
            print(f"Skipping model '{mid}' (no MOJO?): {e}")
            continue
    raise RuntimeError("No MOJO-capable model found on the leaderboard.")

def main():
    random.seed(42)
    print("Generating training CSV...")
    generate_csv(TRAIN_CSV)

    print("Starting H2O...")
    h2o.init()

    print("Loading data into H2O...")
    df = h2o.import_file(str(TRAIN_CSV))

    # Lab features/target
    x = ["length", "entropy"]
    y = "class"
    df[y] = df[y].asfactor()

    print("Running AutoML...")
    aml = H2OAutoML(
        max_models=20,
        max_runtime_secs=120,
        seed=1,
        sort_metric="AUC"
    )
    aml.train(x=x, y=y, training_frame=df)

    print("\n=== Leaderboard (top 10) ===")
    print(aml.leaderboard.head(rows=10))

    # Export a MOJO for production scoring
    mojo_out = MODEL_DIR / "DGA_Leader.zip"
    export_mojo_from_leaderboard(aml, mojo_out)

    # Save a copy of leaderboard for reference
    lb_csv = MODEL_DIR / "leaderboard.csv"
    h2o.as_list(aml.leaderboard).to_csv(lb_csv, index=False)
    print(f"Saved leaderboard to: {lb_csv}")

    h2o.shutdown(prompt=False)
    print("Done.")

if __name__ == "__main__":
    main()

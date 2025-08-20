# Manual Verification

1. **Train and export the model (optional if `models/DGA_Leader.zip` already exists):**
   ```bash
   python 1_train_and_export.py
   ```
2. **Legit domain** – should return `legit` and no playbook:
   ```bash
   python 2_analyze_domain.py --domain google.com
   ```
3. **DGA-like domain** – should return `dga` and print a 3–4 step playbook:
   ```bash
   python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
   ```
   The generated instructions are also saved to `prescriptive_playbook.txt`.
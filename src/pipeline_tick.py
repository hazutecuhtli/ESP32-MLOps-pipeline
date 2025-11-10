# src/pipeline_tick.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]   # Root path
DATA = ROOT / "src\data"
FLAG = DATA / "RETRAIN_NEEDED.flag"
HIST = DATA / "monitor_history.parquet"

# ******************************************************************************************************
# Functions
# ******************************************************************************************************

# Runs CLI commang on the script
def run_cmd(argv, cwd=None):

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # Windows-friendly
    print(">", " ".join(argv))
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=env)
    if r.stdout: print(r.stdout.strip())
    if r.stderr: print(r.stderr.strip())
    return r.returncode == 0

# Append data history for saving
def append_history(**kwargs):
    os.makedirs(DATA, exist_ok=True)
    row = {"_timestamp": pd.Timestamp(datetime.now(timezone.utc)), **kwargs}
    if HIST.exists():
        prev = pd.read_parquet(HIST)
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_parquet(HIST, index=False)
    else:
        pd.DataFrame([row]).to_parquet(HIST, index=False)
    print("Updated monitor historical data.")

# Main
def main():
    
    py = sys.executable  # current Python interpreter
    
    # Retrieve last 12h of data (not used during training)
    ok_data = run_cmd([py, "src/data.py", "--hours_start", "12", "--hours_stop", "0", "--name", "pred"], cwd=ROOT)
    if not ok_data:
        append_history(event="tick", stage="data", retrained=False, notes="data_failed")
        return
    
    # Generate features for monitoring
    ok_feat = run_cmd([py, "src/features.py", "--name", "pred"], cwd=ROOT)
    if not ok_feat:
        append_history(event="tick", stage="features", retrained=False, notes="features_failed")
        return

    # Defining CLI parameters for predict and monitoring (topic, local_dir, input_parquet, run_name)
    evals = [
        ("kitchen", r"src\models\kitchen\ols", r"src\data\feat_kit_pred.parquet", "inference_kitchen_ols"),
        ("office",  r"src\models\office\ols",  r"src\data\feat_off_pred.parquet", "inference_office_ols")]

    # Predicting tempreatures for MLOps
    for topic, local_dir, parquet, run_name in evals:
        # 1) PREDICT
        ok_pred = run_cmd(
            [py, "src/predict.py",
             "--local-dir", local_dir,
             "--input-parquet", parquet,
             "--run-name", run_name,
             "--topic", topic,
             "--log-mlflow"],
            cwd=ROOT
        )
        if not ok_pred:
            append_history(event="tick", topic=topic, retrained=False, notes="predict_failed")
            continue

        # Monitoring model performances
        ok_mon = run_cmd(
            [py, "src/monitor.py",
             "--topic", topic,
             "--run-name-like", run_name],
            cwd=ROOT
        )
        if not ok_mon:
            append_history(event="monitor_error", topic=topic, retrained=False, notes="monitor_failed")
            continue

        # Retraining model if performance not good enough
        if FLAG.exists():
            print("Retraining needed. Runing train.py …")
            try:
                reasons = FLAG.read_text().strip()
            except Exception:
                reasons = ""
            
            ok_train = run_cmd([py, "src/train.py"], cwd=ROOT)
            append_history(
                event="tick", topic=topic,
                retrained=bool(ok_train),
                notes=("retrained" if ok_train else "train_failed"),
            )
            if ok_train:
                FLAG.unlink(missing_ok=True)
            if reasons:
                print("Reasons:", reasons)
        else:
            append_history(event="tick", topic=topic, retrained=False, notes="no_retrain_needed")

# ******************************************************************************************************
# Main
# ******************************************************************************************************

if __name__ == "__main__":
    main()


# ******************************************************************************************************
# Fin
# ******************************************************************************************************

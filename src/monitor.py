# src/monitor.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
import os
import numpy as np
import pandas as pd
import mlflow
# al inicio del archivo
import argparse
from pathlib import Path
# ******************************************************************************************************
# Config
# ******************************************************************************************************
# Defining paths
mlruns_dir = os.path.join(os.getcwd(), 'src', 'mlruns')
PRED_LOG   = r"src\data\predictions_log.parquet" 
TRUTH_FILE = r"src\data\truth_log.parquet"       
# Defining threshold values
THRESHOLDS = {
    "mae_12h": 0.6,        
    "ewma_alpha": 0.2,     
    "ewma_limit": 0.7,     
    "zscore_limit": 3.0,   
    "zscore_features_needed": 2}

# ******************************************************************************************************
# Functions
# ******************************************************************************************************

# Calculating the Exponentially Weighted Moving Average of Absolute Error
def ewma_abs_error(errors: pd.Series, alpha: float) -> float:
    ewma = None
    for e in errors.abs().dropna():
        ewma = e if ewma is None else alpha * e + (1 - alpha) * ewma
    return float(ewma) if ewma is not None else np.nan

# Function to ensure ground truth availability
def load_truth_with_fallback(df_pred: pd.DataFrame, truth_path: str) -> pd.DataFrame:
    # Try loading ground truth from file
    if os.path.exists(truth_path):
        df_t = pd.read_parquet(truth_path)
        if not df_t.empty and {"_time","topic","temp_c"}.issubset(df_t.columns):
            df_t = df_t[["_time","topic","temp_c"]].dropna()
            return df_t
    # Using 'target' column from prediction logs
    if "target" in df_pred.columns and df_pred["target"].notna().any():
        return (df_pred.rename(columns={"target": "temp_c"})
                      .loc[:, ["_time","topic","temp_c"]]
                      .dropna())
    # Not ground truth available
    return pd.DataFrame()

# Retrieving arguments from CLI
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, default=None, help="Filtra por topic (e.g., kitchen u office)")
    ap.add_argument("--model-uri-like", type=str, default=None, help="Substring para filtrar model_uri")
    ap.add_argument("--run-name-like", type=str, default=None, help="Substring para filtrar run_name")
    return ap.parse_args()

# ******************************************************************************************************
# Main
# ******************************************************************************************************

def main():

    # MLflow
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    mlflow.set_experiment("esp32_forecast_monitor")

    # Loading predictions
    if not os.path.exists(PRED_LOG):
        print("No hay predicciones aún.")
        return
    df_pred = pd.read_parquet(PRED_LOG)
    if df_pred.empty:
        print("No hay predicciones aún.")
        return

    # Retrieving arguments from CLI
    args = parse_args()

    # Filtering using args from CLI
    if args.topic and "topic" in df_pred.columns:
        df_pred = df_pred[df_pred["topic"] == args.topic]

    if args.model_uri_like and "model_uri" in df_pred.columns:
        df_pred = df_pred[df_pred["model_uri"].astype(str).str.contains(args.model_uri_like, na=False)]

    if args.run_name_like and "run_name" in df_pred.columns:
        df_pred = df_pred[df_pred["run_name"].astype(str).str.contains(args.run_name_like, na=False)]


    # Minimugn data structure
    required = {"_time","topic","yhat"}
    if not required.issubset(df_pred.columns):
        print("predictions_log.parquet no tiene columnas mínimas [_time, topic, yhat].")
        return

    # Normalizing data
    df_pred["_time"] = pd.to_datetime(df_pred["_time"], utc=True, errors="coerce")
    df_pred = df_pred.dropna(subset=["_time","topic","yhat"])

    # Using last context
    def last_non_null(df, col):
        return df[col].dropna().iloc[-1] if col in df.columns and df[col].notna().any() else None
    latest_model = last_non_null(df_pred, "model_uri")
    latest_run   = last_non_null(df_pred, "run_name")
    latest_topic = last_non_null(df_pred, "topic")

    if latest_model is not None:
        df_pred = df_pred[df_pred["model_uri"] == latest_model]
    if latest_run is not None:
        df_pred = df_pred[df_pred["run_name"] == latest_run]
    if latest_topic is not None:
        df_pred = df_pred[df_pred["topic"] == latest_topic]


    # If explicit filter weres used on the input arguments
    if not (args.topic or args.model_uri_like or args.run_name_like):
        latest_model = last_non_null(df_pred, "model_uri")
        latest_run   = last_non_null(df_pred, "run_name")
        latest_topic = last_non_null(df_pred, "topic")

        if latest_model is not None:
            df_pred = df_pred[df_pred["model_uri"] == latest_model]
        if latest_run is not None:
            df_pred = df_pred[df_pred["run_name"] == latest_run]
        if latest_topic is not None:
            df_pred = df_pred[df_pred["topic"] == latest_topic]

        print(f"Evaluating monitor (last context): model_uri={latest_model or 'unknown'} | run_name={latest_run or 'unknown'} | topic={latest_topic or 'unknown'}")
    else:
        # Printing filteres results
        model_preview = df_pred["model_uri"].dropna().iloc[-1] if "model_uri" in df_pred.columns and df_pred["model_uri"].notna().any() else "unknown"
        run_preview   = df_pred["run_name"].dropna().iloc[-1]  if "run_name"  in df_pred.columns and df_pred["run_name"].notna().any() else "unknown"
        topic_preview = df_pred["topic"].dropna().iloc[-1]     if df_pred["topic"].notna().any() else "unknown"
        print(f"Evaluating monitor (CLI filters): model_uri~{args.model_uri_like or '*'} | run_name~{args.run_name_like or '*'} | topic={args.topic or '*'}")

    
    # Using only the last prediction
    df_pred = df_pred.sort_values("_time").drop_duplicates(subset=["_time","topic"], keep="last")

    print(f"Evaluating monitor: model_uri={latest_model or 'unknown'} | run_name={latest_run or 'unknown'} | topic={latest_topic or 'unknown'}")

    # Loasing ground truth
    df_truth = load_truth_with_fallback(df_pred, TRUTH_FILE)
    if df_truth.empty:
        print("Not ground truth available")
        return
    df_truth["_time"] = pd.to_datetime(df_truth["_time"], utc=True, errors="coerce")
    df_truth = df_truth.dropna(subset=["_time","topic","temp_c"])

    # Creating data for monitoring
    df = pd.merge(df_pred, df_truth, on=["_time","topic"], how="inner", suffixes=("", "_real"))
    df = df.sort_values("_time").dropna(subset=["yhat","temp_c"])
    if df.empty:
        print("Not prediction/truth cases availables yet")
        return


    ## 2) Asegurar tipos y orden antes de métricas
    df["_time"]  = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["yhat"]   = pd.to_numeric(df["yhat"], errors="coerce")
    df["temp_c"] = pd.to_numeric(df["target"], errors="coerce")

    # Errors and temporal index
    df["error"] = df["temp_c"] - df["yhat"]
    df = df.set_index("_time").sort_index()

    roll_mae_4h   = df.iloc[-4:]["error"].abs().mean()  
    roll_rmse_4h  = np.sqrt((df.iloc[-4:]["error"]**2).mean())   
    roll_mae_12h  = df.iloc[-12:]["error"].abs().mean()
    roll_rmse_12h = np.sqrt((df.iloc[-12:]["error"]**2).mean()) 

    # EWMA del |error|
    ewma = ewma_abs_error(df["error"], THRESHOLDS["ewma_alpha"])
    
    # Drift simple de features, determine data drift
    feature_cols = [c for c in df.columns if c.startswith(("lag_", "hum_lag_"))]
    drift_flags = 0
    if feature_cols:
        feat = df[feature_cols]
        ref = feat[-12:]
        cur = feat[-4:]
        if len(ref) > 1 and len(cur) > 0:
            z = ((cur.mean() - ref.mean()) / ref.std(ddof=1).replace(0, np.nan)).abs().fillna(0)
            drift_flags = int((z > THRESHOLDS["zscore_limit"]).sum())        

    # Retraining thresholds evaluation
    retrain = False
    reasons = []
    if pd.notna(roll_mae_12h) and roll_mae_12h > THRESHOLDS["mae_12h"]:
        retrain = True; reasons.append(f"MAE_12h={roll_mae_12h:.3f} > {THRESHOLDS['mae_12h']}")
    if pd.notna(ewma) and ewma > THRESHOLDS["ewma_limit"]:
        retrain = True; reasons.append(f"EWMA|error|={ewma:.3f} > {THRESHOLDS['ewma_limit']}")
    if feature_cols and drift_flags >= THRESHOLDS["zscore_features_needed"]:
        retrain = True; reasons.append(f"Drift en {drift_flags} features (z>{THRESHOLDS['zscore_limit']})")

    # MLflow log
    with mlflow.start_run(run_name=f"monitor_{latest_run or 'unknown'}"):
        def _f(x): return float(x) if (x is not None and not pd.isna(x)) else np.nan
        mlflow.log_param("model_uri", latest_model or "unknown")
        mlflow.log_param("run_name",  latest_run  or "unknown")
        mlflow.log_param("topic",     latest_topic or "unknown")
        mlflow.log_metric("mae_4h",   _f(roll_mae_4h))
        mlflow.log_metric("rmse_4h",  _f(roll_rmse_4h))
        mlflow.log_metric("mae_12h",  _f(roll_mae_12h))
        mlflow.log_metric("rmse_12h", _f(roll_rmse_12h))
        mlflow.log_metric("ewma_abs_error", _f(ewma))
        mlflow.set_tags({"retrain_needed": str(retrain), "reasons": "; ".join(reasons)})

    metrics_mon = {
        "mae_4h":roll_mae_4h,
        "rmse_4h":roll_rmse_4h,
        "mae_12h":roll_mae_12h,
        "rmse_12h":roll_rmse_12h,
        "ewma_abs_error":ewma}

    print(metrics_mon)



    # Local Flag
    FLAG = r"data\RETRAIN_NEEDED.flag"
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "src" / "data"
    FLAG = DATA / "RETRAIN_NEEDED.flag"    
    if retrain:
        DATA.mkdir(parents=True, exist_ok=True)
        FLAG.write_text("\n".join(reasons))
        print("RETRAIN_NEEDED:", reasons)
    else:
        if FLAG.exists():
            FLAG.unlink()
        print("*****-----> Modelo OK. No retraining needed <-----*****\n")

    # Saving locally
    HIST = r"src\data\monitor_history.parquet"
    row = {
        "_timestamp": pd.Timestamp.utcnow(),
        "model_uri":  latest_model or "unknown",
        "run_name":   latest_run or "unknown",
        "topic":      latest_topic or "unknown",
        "mae_4h":     float(roll_mae_4h)   if pd.notna(roll_mae_4h) else np.nan,
        "rmse_4h":    float(roll_rmse_4h)  if pd.notna(roll_rmse_4h) else np.nan,
        "mae_12h":    float(roll_mae_12h)  if pd.notna(roll_mae_12h) else np.nan,
        "rmse_12h":   float(roll_rmse_12h) if pd.notna(roll_rmse_12h) else np.nan,
        "ewma_abs":   float(ewma)          if pd.notna(ewma) else np.nan,
        "drift_flags": int(drift_flags),
        "retrain_needed": bool(retrain),
        "reasons":    "; ".join(reasons),
    }
    os.makedirs("data", exist_ok=True)
    if os.path.exists(HIST):
        prev = pd.read_parquet(HIST)
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_parquet(HIST, index=False)
    else:
        pd.DataFrame([row]).to_parquet(HIST, index=False)
        

if __name__ == "__main__":
    main()


# ******************************************************************************************************
# Fin
# ******************************************************************************************************

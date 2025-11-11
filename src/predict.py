# predict.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
import os
import json, argparse, time
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
from pathlib import Path
from datetime import datetime, timezone

# ******************************************************************************************************
# Defining Classes
# ******************************************************************************************************

# Fallback local (If not pyfunc is available

class LocalOLSHAC:
    def __init__(self, dirpath: str):
        self.scaler = joblib.load(os.path.join(dirpath, "scaler.pkl"))
        self.coef   = np.load(os.path.join(dirpath, "coefficients.npy"))
        with open(os.path.join(dirpath, "meta.json"), "r") as f:
            self.meta = json.load(f)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # Defining data normalization and structure for predictions
        feats = self.meta["columns"][1:]      # exclude "const"
        df_feats = df[feats].copy()
        Xs = self.scaler.transform(df_feats)  
        Xc = np.c_[np.ones((Xs.shape[0], 1)), Xs]   # add intercept
        yhat = Xc @ self.coef
        return pd.Series(yhat, name="prediction", index=df.index)

# ******************************************************************************************************
# Functions
# ******************************************************************************************************

# Loading encapsulated model
def load_model(model_uri: str = None, local_dir: str = None):
    if model_uri:
        return mlflow.pyfunc.load_model(model_uri)  # objeto con .predict(df)
    if local_dir:
        return LocalOLSHAC(local_dir)
    raise ValueError("Provee --model-uri o --local-dir")

# Defining models performance metrics
def compute_live_metrics(df: pd.DataFrame, yhat: pd.Series, target_col="target"):
    if target_col in df.columns:
        y = df[target_col].to_numpy().ravel()
        e = y - yhat.to_numpy().ravel()
        mae  = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e * e)))
        ss_res = float(np.sum(e * e))
        ss_tot = float(np.sum((y - np.mean(y))**2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        return {"mae_live": mae, "rmse_live": rmse, "r2_live": r2}
    return {}

# Genreating data log
def append_predictions_log(
    df_out: pd.DataFrame,
    pred_col: str = "yhat",
    path: str = r"src\data\predictions_log.parquet",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    log = pd.DataFrame(index=df_out.index)

    # Creating dataframe with log data
    model_uri = getattr(df_out, "model_uri", None) or "unknown"
    run_name  = getattr(df_out, "run_name",  None) or "unknown"
    log["model_uri"] = model_uri
    log["run_name"]  = run_name

    # timestamp
    if "_time" in df_out.columns:
        log["_time"] = pd.to_datetime(df_out["_time"], utc=True, errors="coerce")
    else:
        log["_time"] = pd.Timestamp(datetime.now(timezone.utc))

    # topic/sensor
    log["topic"] = df_out["topic"] if "topic" in df_out.columns else "unknown"

    # yhat y and target
    log["yhat"] = df_out[pred_col]
    if "target" in df_out.columns:
        log["target"] = df_out["target"]

    # features for drift
    extra = [c for c in df_out.columns if c.startswith(("lag_", "hum_lag_", "temp_", "hum_"))]
    if extra:
        log = pd.concat([log, df_out[extra]], axis=1)

    # append to create data log
    if os.path.exists(path):
        prev = pd.read_parquet(path)
        pd.concat([prev, log], ignore_index=True).to_parquet(path, index=False)
    else:
        log.to_parquet(path, index=False)

    print(f"Registro de predicciones actualizado: {path}")

# Predicting temperatures
def run_once(args):
    
    # MLflow local
    MLRUNS_DIR = Path(args.tracking_dir)
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment(args.experiment)

    # Loading
    df = pd.read_parquet(args.input_parquet)
    model = load_model(args.model_uri, args.local_dir)

    # Prediction
    yhat = model.predict(df)
    if not isinstance(yhat, pd.Series):
        yhat = pd.Series(np.asarray(yhat).ravel(), name="prediction", index=df.index)
    # Output
    out_path = os.path.splitext(args.input_parquet)[0] + "_res.parquet"
    df_out = df.copy()
    df_out["yhat"] = yhat

    # Model context
    df_out.model_uri = args.model_uri or args.local_dir
    df_out.run_name  = args.run_name

    # CLI inputs
    if args.topic:
        df_out["topic"] = args.topic
    elif "topic" not in df_out.columns:
        guess = None
        try:
            parts = set(map(str.lower, Path(args.local_dir or "").parts))
            if "kitchen" in parts: guess = "kitchen"
            if "office"  in parts: guess = "office"
        except Exception:
            pass
        df_out["topic"] = guess or "unknown"

    # Saving predictions
    df_out.to_parquet(out_path, index=False)
    print(f"Predicciones guardadas en: {out_path}")

    # Performance metrics
    metrics = compute_live_metrics(df_out, df_out["yhat"], target_col="target")
    if metrics:
        print("Live metrics:", metrics)

        print()
        print('     Live Performance     ')
        print(f'-------------------------')
        print(f'|  Metric   |   Value   |')
        print(f'-------------------------')
        print(f"|  mae      |  {metrics['mae_live']:.4f}   |")
        print(f"|  rmse     |  {metrics['rmse_live']:.4f}   |")
        print(f"|  r2       |  {metrics['r2_live']:.4f}   |")
        print()
        
    # Adding data log
    append_predictions_log(df_out, pred_col="yhat", path=args.pred_log)

    # MLflow data for inference
    if args.log_mlflow:
        with mlflow.start_run(run_name=args.run_name):
            # Input features structure
            drop_cols = [c for c in ["target", "yhat"] if c in df.columns]
            n_features = df.drop(columns=drop_cols, errors="ignore").shape[1]
            mlflow.log_params({
                "model_uri": args.model_uri or "",
                "local_dir": args.local_dir or "",
                "input_parquet": args.input_parquet,
                "n_rows": len(df),
                "n_features": n_features,
            })
            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
            mlflow.log_artifact(out_path, artifact_path="inference")

# Main
def main():

    track_uri = os.path.join(os.getcwd(), 'src', 'mlruns')

    # Getting parameters from CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", type=str, default=None,
                    help="MLflow URI: runs:/<run_id>/model o models:/nombre/Production")
    ap.add_argument("--local-dir", type=str, default=None,
                    help="Carpeta con scaler.pkl, coefficients.npy, meta.json")
    ap.add_argument("--input-parquet", type=str, required=True,
                    help="Parquet con features (y opcionalmente 'target', '_time', 'topic')")
    ap.add_argument("--log-mlflow", action="store_true",
                    help="Loggear métricas/artefactos de inferencia en MLflow")
    ap.add_argument("--run-name", type=str, default="inference_office_ols")
    ap.add_argument("--tracking-dir", type=str,
                    default=track_uri, 
                    help="Carpeta local donde MLflow guardará runs/artifacts")
    ap.add_argument("--experiment", type=str, default="esp32_forecast_inference",
                    help="Nombre del experimento MLflow")
    ap.add_argument("--pred-log", type=str, default=r"src\data\predictions_log.parquet",
                    help="Archivo parquet acumulado para el monitor")
    ap.add_argument("--stream", action="store_true", help="Corre en bucle (modo streaming)")
    ap.add_argument("--interval", type=int, default=60, help="Segundos entre corridas en modo streaming")
    ap.add_argument("--topic", type=str, default=None, help="Nombre del sensor/ubicación (p.ej. kitchen, office)")

    args = ap.parse_args()

    if not args.model_uri and not args.local_dir:
        raise SystemExit("Debes pasar --model-uri o --local-dir")

    if not args.stream:
        run_once(args)
    else:
        print(f"Modo streaming ON — cada {args.interval}s")
        while True:
            try:
                run_once(args)
            except Exception as e:
                print("Error en ciclo de inferencia:", repr(e))
            time.sleep(args.interval)

# ******************************************************************************************************
# Main
# ******************************************************************************************************

if __name__ == "__main__":
    main()

# src/train.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
import os
from pathlib import Path
import mlflow
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc        
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_mlflow
import joblib
import tempfile, shutil
# Defining paths for model monitoring
mlruns_dir = os.path.join(os.getcwd(), 'src', 'mlruns')
mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
mlflow.set_experiment("esp32_forecast")
# ******************************************************************************************************
# Defining Classes
# ******************************************************************************************************

# Encapsulating OLS models using pyfunc for MLFlow monitorin

class OLSHACModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Saving model artifacts
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.coef   = np.load(context.artifacts["coeffs"])
        with open(context.artifacts["meta"], "r") as f:
            self.meta = json.load(f)  # ["columns"] = ["const", feat1, feat2, ...]

    # Defining data workflow for correct predictions
    def predict(self, context, model_input: pd.DataFrame):
        # Defining data normalization and structure for predictions
        feats = self.meta["columns"][1:]      # exclude "const"
        X = model_input[feats].to_numpy()
        Xs = self.scaler.transform(X)
        Xc = np.c_[np.ones((Xs.shape[0], 1)), Xs]   # add intercept
        yhat = Xc @ self.coef
        return pd.Series(yhat, name="prediction")

# ******************************************************************************************************
# Features
# ******************************************************************************************************

# Function that checks if a path really exists
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# Generating OLS model using statmoodels
def _fit_ols_hac(X_train, y_train, X_test, hac_maxlags=4):
    # Add constant
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c  = sm.add_constant(X_test,  has_constant="add")

    model = sm.OLS(y_train, X_train_c)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})  #maxlags number of lags to gnerate features

    y_pred = results.predict(X_test_c)
    return results, y_pred

# Evaluating the performanfe of the OLS model
def _eval(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Genearting the OLS model
def train_model():
    
    # Loading feeatures for generating the model
    df_kitchen = pd.read_parquet("src/data/feat_kit.parquet")
    df_office  = pd.read_parquet("src/data/feat_off.parquet")
    target = "target"
    
    # Creating features datasets
    features_kit = [c for c in df_kitchen.columns if c != target]
    features_off = [c for c in df_office.columns  if c != target]

    # Defining model characeristsics and datasets sizes
    test_size   = 0.20
    hac_maxlags = 4
    scale_here  = True

    # Creating training and testing datasets
    X_train_kit, X_test_kit, y_train_kit, y_test_kit = train_test_split(
        df_kitchen[features_kit], df_kitchen[target], test_size=test_size, shuffle=False
    )
    X_train_off, X_test_off, y_train_off, y_test_off = train_test_split(
        df_office[features_off], df_office[target], test_size=test_size, shuffle=False
    )

    # Normalizing data to reduce bias (Scaling)
    scaler_kit = StandardScaler()
    scaler_off = StandardScaler()

    # Generating Kitchen OLS model
    X_train_kit_s = scaler_kit.fit_transform(X_train_kit) if scale_here else X_train_kit.values
    X_test_kit_s  = scaler_kit.transform(X_test_kit)     if scale_here else X_test_kit.values
    # Generating Office OLS model
    X_train_off_s = scaler_off.fit_transform(X_train_off) if scale_here else X_train_off.values
    X_test_off_s  = scaler_off.transform(X_test_off)      if scale_here else X_test_off.values

    # Creating predictions and target to vectors for evaluatio of performances
    y_train_kit_1d = y_train_kit.to_numpy().ravel()
    y_test_kit_1d  = y_test_kit.to_numpy().ravel()
    y_train_off_1d = y_train_off.to_numpy().ravel()
    y_test_off_1d  = y_test_off.to_numpy().ravel()

    # Defining saving paths
    out_dir_kit = _ensure_dir(os.path.join("src", "models", "kitchen", "ols"))
    out_dir_off = _ensure_dir(os.path.join("src", "models", "office",  "ols"))

    # Saving Models performances on Mlfow
    # ------------------------------------------------------------
    # KITCHEN
    # ------------------------------------------------------------
    
    with mlflow.start_run(run_name="kitchen_ols", nested=True):

        # Genrating data to be monitored on mlflow
        results_kit, y_pred_kit = _fit_ols_hac(X_train_kit_s, y_train_kit_1d, X_test_kit_s, hac_maxlags)
        mae_k, rmse_k, r2_k = _eval(y_test_kit_1d, y_pred_kit)

        # Saving
        joblib.dump(scaler_kit, os.path.join(out_dir_kit, "scaler.pkl"))
        np.save(os.path.join(out_dir_kit, "coefficients.npy"), results_kit.params)

        columns_kit = ["const"] + list(features_kit)
        meta_k = {
            "topic": "kitchen",
            "model": "ols_hac",
            "columns": columns_kit,
            "hac_maxlags": hac_maxlags,
            "metrics": {"mae": float(mae_k), "rmse": float(rmse_k), "r2": float(r2_k)}
        }
        with open(os.path.join(out_dir_kit, "meta.json"), "w") as f:
            json.dump(meta_k, f, indent=2)

        pd.DataFrame({"y": y_test_kit_1d, "yhat": y_pred_kit}).to_parquet(
            os.path.join(out_dir_kit, "pred.parquet"), index=False
        )

        # MLflow logs
        mlflow.set_tags({"project": "esp32-mlops", "topic": "kitchen", "model": "ols_hac"})
        mlflow.log_params({
            "features": len(features_kit),
            "test_size": test_size,
            "hac_maxlags": hac_maxlags,
            "scaled_here": scale_here
        })
        mlflow.log_metric("mae",  mae_k)
        mlflow.log_metric("rmse", rmse_k)
        mlflow.log_metric("r2",   r2_k)
        if getattr(results_kit, "fvalue", None) is not None:
            mlflow.log_metric("Fstat", float(results_kit.fvalue))
        if getattr(results_kit, "f_pvalue", None) is not None:
            mlflow.log_metric("Fpval", float(results_kit.f_pvalue))

        # Model artifacts
        mlflow.log_artifacts(out_dir_kit, artifact_path="kitchen_ols")

        # Mlflow encapsulation
        artifacts_k = {
            "scaler": os.path.join(out_dir_kit, "scaler.pkl"),
            "coeffs": os.path.join(out_dir_kit, "coefficients.npy"),
            "meta":   os.path.join(out_dir_kit, "meta.json"),
        }

        input_example_k = X_test_kit.iloc[:5].copy()  # Same columns as in training
        signature_k = mlflow.models.infer_signature(
            model_input=X_test_kit,
            model_output=pd.Series(np.zeros(len(input_example_k)), name="prediction")
        )

        tmpdir = tempfile.mkdtemp()   # Temporal folder
        try:
            mlflow.pyfunc.save_model(
                path=tmpdir,            
                python_model=OLSHACModel(),
                artifacts=artifacts_k,
                signature=signature_k,
                input_example=input_example_k
            )
            mlflow.log_artifacts(tmpdir, artifact_path="model")  # Storing data on mflow
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)  # Deleting temp folder

        # URI del modelo de este run
        run_k = mlflow.active_run()

    # ------------------------------------------------------------
    # OFFICE
    # ------------------------------------------------------------

    # Genrating data to be monitored on mlflow
    with mlflow.start_run(run_name="office_ols", nested=True):
        results_off, y_pred_off = _fit_ols_hac(X_train_off_s, y_train_off_1d, X_test_off_s, hac_maxlags)
        mae_o, rmse_o, r2_o = _eval(y_test_off_1d, y_pred_off)

        # Saving
        joblib.dump(scaler_off, os.path.join(out_dir_off, "scaler.pkl"))
        np.save(os.path.join(out_dir_off, "coefficients.npy"), results_off.params)

        columns_off = ["const"] + list(features_off)
        meta_o = {
            "topic": "office",
            "model": "ols_hac",
            "columns": columns_off,
            "hac_maxlags": hac_maxlags,
            "metrics": {"mae": float(mae_o), "rmse": float(rmse_o), "r2": float(r2_o)}
        }
        with open(os.path.join(out_dir_off, "meta.json"), "w") as f:
            json.dump(meta_o, f, indent=2)

        pd.DataFrame({"y": y_test_off_1d, "yhat": y_pred_off}).to_parquet(
            os.path.join(out_dir_off, "pred.parquet"), index=False
        )

        # MLflow logs
        mlflow.set_tags({"project": "esp32-mlops", "topic": "office", "model": "ols_hac"})
        mlflow.log_params({
            "features": len(features_off),
            "test_size": test_size,
            "hac_maxlags": hac_maxlags,
            "scaled_here": scale_here
        })
        mlflow.log_metric("mae",  mae_o)
        mlflow.log_metric("rmse", rmse_o)
        mlflow.log_metric("r2",   r2_o)
        if getattr(results_off, "fvalue", None) is not None:
            mlflow.log_metric("Fstat", float(results_off.fvalue))
        if getattr(results_off, "f_pvalue", None) is not None:
            mlflow.log_metric("Fpval", float(results_off.f_pvalue))

        # Model artifacts
        mlflow.log_artifacts(out_dir_off, artifact_path="office_ols")

        # Mlflow encapsulation
        artifacts_o = {
            "scaler": os.path.join(out_dir_off, "scaler.pkl"),
            "coeffs": os.path.join(out_dir_off, "coefficients.npy"),
            "meta":   os.path.join(out_dir_off, "meta.json"),
        }

        input_example_o = X_test_off.iloc[:5].copy()
        signature_o = mlflow.models.infer_signature(
            model_input=X_test_off,
            model_output=pd.Series(np.zeros(len(input_example_o)), name="prediction")
        )

        tmpdir = tempfile.mkdtemp()   # ← Making a tempfolder
        try:
            mlflow.pyfunc.save_model(
                path=tmpdir,                  # guarda el modelo allí
                python_model=OLSHACModel(),
                artifacts=artifacts_k,
                signature=signature_k,
                input_example=input_example_k
            )
            mlflow.log_artifacts(tmpdir, artifact_path="model")  # sube la carpeta a MLflow
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)  # borra el tmpdir

        # URI del modelo de este run
        run_o = mlflow.active_run()

    # Summary - Training performance
    print('\n')
    print(f"-----> Kitchen — MAE={mae_k:.3f} | RMSE={rmse_k:.3f} | R2={r2_k:.3f} <-----")
    print(f"-----> Office  — MAE={mae_o:.3f} | RMSE={rmse_o:.3f} | R2={r2_o:.3f} <-----")
    
# ******************************************************************************************************
# Main
# ******************************************************************************************************

if __name__ == "__main__":

   # Training model
    train_model()

# ******************************************************************************************************
# Fin
# ******************************************************************************************************

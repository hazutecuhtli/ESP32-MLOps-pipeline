# src/utils.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
import os
import json
import mlflow
from dotenv import load_dotenv
load_dotenv()

DEFAULT_TZ = os.getenv("LOCAL_TZ", "America/Merida")  # Usa Merida por defecto

# ******************************************************************************************************
# Functions
# ******************************************************************************************************

# Inicializa MLflow usando variables de entorno (URI y Experimento)
def load_mlflow():
    """
    Inicializa MLflow leyendo variables de entorno si existen.
    - MLFLOW_TRACKING_URI (e.g. http://<ip>:5000)
    - MLFLOW_EXPERIMENT  (e.g. esp32_forecast)
    """
    tracking_uri = os.getenv("http://localhost:5000", "http://localhost:5000")  
    experiment   = os.getenv("MLFLOW_EXPERIMENT", "esp32_forecast")
    print(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"✅ MLflow — URI: {tracking_uri} | Experiment: {experiment}")

# Crea el directorio si no existe y devuelve la ruta
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# Guarda un diccionario en archivo JSON
def save_json(data: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# Loguea métricas numéricas en MLflow
def log_metrics_mlflow(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))

# Retorna el timezone local del proyecto
def local_tz() -> str:
    """Devuelve el timezone local del proyecto (por defecto America/Merida)."""
    return DEFAULT_TZ

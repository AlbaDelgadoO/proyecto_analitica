import bentoml
import numpy as np
import json
import logging
from bentoml.io import JSON
from sklearn.preprocessing import StandardScaler

# --- Configuración de logs ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bentoml.service")

# ============================================================
# 1. CARGA DE CONFIGURACIÓN (TOP FEATURES)
# ============================================================
try:
    with open("DatasetProcesado/top_features.json", "r") as f:
        TOP_FEATURES_LIST = json.load(f)
except FileNotFoundError:
    logger.error("ERROR: No se encontró DatasetProcesado/top_features.json")
    TOP_FEATURES_LIST = []

if not TOP_FEATURES_LIST:
    raise ValueError("❌ top_features.json está vacío. No se puede continuar.")
logger.info(f"✅ Cargadas {len(TOP_FEATURES_LIST)} features del JSON.")

# Crear un diccionario para mapear cada feature a su índice
FEATURE_INDICES = {feature: idx for idx, feature in enumerate(TOP_FEATURES_LIST)}

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================
def build_scaler(model_ref):
    """Reconstruye el StandardScaler a partir de metadatos del modelo."""
    try:
        mean = np.array(model_ref.info.metadata["scaler_mean"])
        scale = np.array(model_ref.info.metadata["scaler_scale"])
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = scale ** 2
        return scaler
    except Exception as e:
        logger.error(f"Error reconstruyendo scaler para {model_ref.tag}: {e}")
        return None

def preprocess_input(input_array):
    """Selecciona features y limpia NaNs/Infs."""
    if not isinstance(input_array, np.ndarray):
        input_array = np.array(input_array, dtype=np.float32)

    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    # Seleccionar features
    try:
        indices = list(range(len(TOP_FEATURES_LIST)))
        X = input_array[:, indices].astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X
    except Exception as e:
        logger.error(f"Error en preprocess_input: {e}")
        raise

# ============================================================
# 3. CREAR SERVICIO BENTOML
# ============================================================
svc = bentoml.Service("tep_fault_classifier")

# ============================================================
# 4. CACHE PARA RUNNERS Y SCALERS
# ============================================================
model_cache = {}

def get_model_runner(model_name, module="sklearn"):
    """Carga el modelo y scaler si no están en cache y devuelve ambos."""
    if model_name in model_cache:
        return model_cache[model_name]

    # Cargar modelo
    try:
        if module == "sklearn":
            model_ref = bentoml.sklearn.get(model_name)
        elif module == "tensorflow":
            model_ref = bentoml.tensorflow.get(model_name)
        else:
            raise ValueError(f"Módulo desconocido: {module}")
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo {model_name}: {e}")
        raise

    # Reconstruir scaler
    scaler = build_scaler(model_ref)
    if scaler is None:
        logger.warning(f"Scaler no disponible para {model_name}, se usará input sin escalar.")

    runner = model_ref.to_runner()
    model_cache[model_name] = (runner, scaler)
    return runner, scaler

# ============================================================
# 5. ENDPOINTS CON LOGS DETALLADOS
# ============================================================
async def predict_generic(input_data, model_name, module="sklearn"):
    try:
        logger.info("=== INPUT ORIGINAL ===")
        logger.info(input_data)
        logger.info("Tipo de input: %s", type(input_data))

        X = preprocess_input(input_data)
        logger.info("=== INPUT PROCESADO ===")
        logger.info(X)
        logger.info("Forma: %s", X.shape)

        runner, scaler = get_model_runner(model_name, module)

        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        logger.info("=== X ESCALADO ===")
        logger.info(X_scaled)

        prediction = await runner.predict.async_run(X_scaled)
        logger.info("=== PREDICCIÓN ===")
        logger.info(prediction)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.exception("ERROR en predict_generic:")
        return {"error": str(e)}

@svc.api(input=JSON(), output=JSON())
async def predict_binary(input_data):
    return await predict_generic(input_data, "tep_model1:latest", "sklearn")

@svc.api(input=JSON(), output=JSON())
async def predict_horizon(input_data):
    return await predict_generic(input_data, "tep_model2:latest", "sklearn")

@svc.api(input=JSON(), output=JSON())
async def predict_multiclass(input_data):
    return await predict_generic(input_data, "tep_model3:latest", "sklearn")

@svc.api(input=JSON(), output=JSON())
async def predict_isolation(input_data):
    res = await predict_generic(input_data, "tep_model4:latest", "sklearn")
    # Ajuste para método de aislamiento: convertir -1 a 1
    if "prediction" in res and res["prediction"] == -1:
        res["prediction"] = 1
    else:
        res["prediction"] = 0
    return res

@svc.api(input=JSON(), output=JSON())
async def predict_autoencoder(input_data):
    try:
        X = preprocess_input(input_data)
        runner, scaler = get_model_runner("tep_model5:latest", "tensorflow")

        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        reconstruction = await runner.predict.async_run(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstruction))
        if not np.isfinite(mse):
            mse = 999.0

        ae_threshold = float(
            bentoml.tensorflow.get("tep_model5:latest").info.metadata.get("threshold", 0.0)
        )
        res = 1 if mse > ae_threshold else 0

        logger.info("=== AUTOENCODER ===")
        logger.info("MSE: %f, Threshold: %f, Prediction: %d", mse, ae_threshold, res)
        return {"prediction": res, "mse": float(mse), "threshold": float(ae_threshold)}

    except Exception as e:
        logger.exception("ERROR en predict_autoencoder:")
        return {"error": str(e)}

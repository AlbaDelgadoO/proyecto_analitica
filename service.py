import bentoml
import numpy as np
import json
import logging
from bentoml.io import JSON
from sklearn.preprocessing import StandardScaler

# ---------------------------
# CONFIGURACIÓN DE LOGS
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tep_fault_classifier")

# ---------------------------
# 1. CARGA DE TOP FEATURES
# ---------------------------
try:
    with open("DatasetProcesado/top_features.json", "r") as f:
        TOP_FEATURES = json.load(f)
except FileNotFoundError:
    logger.error("ERROR: No se encontró top_features.json")
    TOP_FEATURES = []

if not TOP_FEATURES:
    raise ValueError("❌ top_features.json está vacío. No se puede continuar.")
logger.info(f"✅ Cargadas {len(TOP_FEATURES)} features del JSON.")

# ---------------------------
# 2. FUNCIONES AUXILIARES
# ---------------------------
def build_scaler(model_ref):
    """Reconstruye StandardScaler desde metadata."""
    try:
        mean = np.array(model_ref.info.metadata.get("scaler_mean", []))
        scale = np.array(model_ref.info.metadata.get("scaler_scale", []))
        if len(mean) == 0 or len(scale) == 0:
            return None
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = scale ** 2
        return scaler
    except Exception as e:
        logger.warning(f"No se pudo reconstruir scaler: {e}")
        return None

def preprocess_input(input_array):
    """Asegura np.ndarray y selecciona solo TOP_FEATURES."""
    if not isinstance(input_array, np.ndarray):
        input_array = np.array(input_array, dtype=np.float32)

    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    if input_array.shape[1] != len(TOP_FEATURES):
        raise ValueError(
            f"❌ Input tiene {input_array.shape[1]} features pero se esperaban {len(TOP_FEATURES)}"
        )

    X = np.nan_to_num(input_array, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# ---------------------------
# 3. CREAR RUNNERS Y ESCALERS
# ---------------------------
runner1 = bentoml.sklearn.get("tep_model1:latest").to_runner()
runner2 = bentoml.sklearn.get("tep_model2:latest").to_runner()
runner3 = bentoml.sklearn.get("tep_model3:latest").to_runner()
runner4 = bentoml.sklearn.get("tep_model4:latest").to_runner()

scaler1 = build_scaler(bentoml.sklearn.get("tep_model1:latest"))
scaler2 = build_scaler(bentoml.sklearn.get("tep_model2:latest"))
scaler3 = build_scaler(bentoml.sklearn.get("tep_model3:latest"))
scaler4 = build_scaler(bentoml.sklearn.get("tep_model4:latest"))

# ---------------------------
# 4. CACHE DE MODELOS
# ---------------------------
model_cache = {
    "tep_model1:latest": (runner1, scaler1),
    "tep_model2:latest": (runner2, scaler2),
    "tep_model3:latest": (runner3, scaler3),
    "tep_model4:latest": (runner4, scaler4),
}

def get_runner_and_scaler(model_name):
    """Devuelve runner y scaler desde cache."""
    return model_cache[model_name]

# ---------------------------
# 5. CREAR SERVICIO
# ---------------------------
svc = bentoml.Service(
    "tep_fault_classifier",
    runners=[runner1, runner2, runner3, runner4]
)

# ---------------------------
# 6. PREDICCIÓN GENÉRICA
# ---------------------------
async def predict_generic(input_data, model_name):
    try:
        logger.info("=== INPUT ORIGINAL ===")
        logger.info(input_data)

        X = preprocess_input(input_data)
        logger.info(f"=== INPUT PROCESADO (shape={X.shape}) ===")
        logger.info(X)

        runner, scaler = get_runner_and_scaler(model_name)
        X_scaled = scaler.transform(X) if scaler else X

        logger.info("=== INPUT ESCALADO ===")
        logger.info(X_scaled)

        prediction = await runner.predict.async_run(X_scaled)
        logger.info(f"=== PREDICCIÓN === {prediction}")

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.exception("ERROR en predict_generic:")
        return {"error": str(e)}

# ---------------------------
# 7. ENDPOINTS
# ---------------------------
@svc.api(input=JSON(), output=JSON())
async def predict_binary(input_data):
    return await predict_generic(input_data, "tep_model1:latest")

@svc.api(input=JSON(), output=JSON())
async def predict_horizon(input_data):
    return await predict_generic(input_data, "tep_model2:latest")

@svc.api(input=JSON(), output=JSON())
async def predict_multiclass(input_data):
    return await predict_generic(input_data, "tep_model3:latest")

@svc.api(input=JSON(), output=JSON())
async def predict_isolation(input_data):
    res = await predict_generic(input_data, "tep_model4:latest")
    if "prediction" in res:
        res["prediction"] = 1 if res["prediction"] == -1 else 0
    return res

import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from sklearn.preprocessing import StandardScaler

# ============================================================
# Función auxiliar: reconstruir scaler desde metadatos
# ============================================================
def build_scaler(model_ref):
    mean = np.array(model_ref.info.metadata["scaler_mean"])
    scale = np.array(model_ref.info.metadata["scaler_scale"])

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2  # requerido por sklearn
    return scaler

# ============================================================
# Cargar modelos desde el Model Store
# ============================================================
model1_ref = bentoml.sklearn.get("tep_model1:latest")
runner1 = model1_ref.to_runner()
features1 = model1_ref.info.metadata["top_features"]
scaler1 = build_scaler(model1_ref)

model2_ref = bentoml.sklearn.get("tep_model2:latest")
runner2 = model2_ref.to_runner()
features2 = model2_ref.info.metadata["top_features"]
scaler2 = build_scaler(model2_ref)

# ============================================================
# Crear servicio BentoML
# ============================================================
svc = bentoml.Service(
    "tep_fault_classifier",
    runners=[runner1, runner2]
)

# ============================================================
# Crear ejemplos de entrada para validación
# ============================================================
sample_input1 = [[0.0] * 10]
sample_input2 = [[0.0] * len(features2)]

# ============================================================
# Endpoint Modelo 1: Clasificación binaria
# ============================================================
@svc.api(input=NumpyNdarray.from_sample(sample_input1), output=NumpyNdarray())
async def predict_model1(input_array: np.ndarray) -> np.ndarray:
    expected = len(features1)

    if input_array.shape[1] != expected:
        raise ValueError(
            f"Modelo 1 espera {expected} features, pero la entrada tiene {input_array.shape[1]}"
        )

    # Escalado consistente con el entrenamiento
    input_scaled = scaler1.transform(input_array)

    prediction = await runner1.predict.async_run(input_scaled)
    return np.array(prediction)

# ============================================================
# Endpoint Modelo 2: Predicción de horizonte de fallo
# ============================================================
@svc.api(input=NumpyNdarray.from_sample(sample_input2), output=NumpyNdarray())
async def predict_model2(input_array: np.ndarray) -> np.ndarray:
    expected = len(features2)

    if input_array.shape[1] != expected:
        raise ValueError(
            f"Modelo 2 espera {expected} features, pero la entrada tiene {input_array.shape[1]}"
        )

    # Escalado consistente con el entrenamiento
    input_scaled = scaler2.transform(input_array)

    prediction = await runner2.predict.async_run(input_scaled)
    return np.array(prediction)
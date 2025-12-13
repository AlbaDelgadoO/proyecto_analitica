import bentoml
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray

# =========================
# Función auxiliar: obtener top-10 features y ejemplo seguro
# =========================
def get_sample_input(model_number=1):
    """
    Devuelve un ejemplo de entrada para BentoML usando las top-10 features
    del modelo seleccionado.
    """
    ranking_file = f"DatasetProcesado/feature_importance_ranking_model{model_number}_top10.csv"
    ranking_top10 = pd.read_csv(ranking_file)
    top_features = ranking_top10['feature'].tolist()

    df = pd.read_csv("DatasetProcesado/TEP_features_train.csv")
    sample_row = df[top_features].iloc[0].fillna(0)  # rellenar NaN con 0
    sample_input = [float(x) for x in sample_row.tolist()]  # convertir a float
    return [sample_input], top_features

# =========================
# Cargar modelos y runners
# =========================
model1_ref = bentoml.sklearn.get("tep_model1:latest")
runner1 = model1_ref.to_runner()

model2_ref = bentoml.sklearn.get("tep_model2:latest")
runner2 = model2_ref.to_runner()

svc = bentoml.Service("tep_fault_classifier", runners=[runner1, runner2])

# =========================
# Valores de ejemplo
# =========================
sample_input1, features1 = get_sample_input(1)
sample_input2, features2 = get_sample_input(2)

# =========================
# Endpoint POST para Modelo 1
# =========================
@svc.api(input=NumpyNdarray.from_sample(sample_input1), output=NumpyNdarray())
async def predict_model1(input_array: np.ndarray) -> np.ndarray:
    """
    Predicción Modelo 1: Clasificación Binaria (Fallo vs Normal)
    input_array: np.ndarray con forma (n_samples, 10 features)
    """
    if input_array.shape[1] != 10:
        raise ValueError(f"Modelo 1 espera 10 features, pero la entrada tiene {input_array.shape[1]}")
    prediction = await runner1.predict.async_run(input_array)
    return np.array(prediction)

# =========================
# Endpoint POST para Modelo 2
# =========================
@svc.api(input=NumpyNdarray.from_sample(sample_input2), output=NumpyNdarray())
async def predict_model2(input_array: np.ndarray) -> np.ndarray:
    """
    Predicción Modelo 2: Predicción de Fallos Futuros (Horizonte)
    input_array: np.ndarray con forma (n_samples, 10 features)
    """
    if input_array.shape[1] != 10:
        raise ValueError(f"Modelo 2 espera 10 features, pero la entrada tiene {input_array.shape[1]}")
    prediction = await runner2.predict.async_run(input_array)
    return np.array(prediction)
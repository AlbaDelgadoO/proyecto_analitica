import bentoml
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Cargar modelo directamente (sin runners)
model = bentoml.sklearn.load_model("tep_random_forest:latest")

# 2. Reconstruir scaler usando el dataset de entrenamiento
df_train = pd.read_csv("DatasetProcesado/TEP_features_train.csv", sep=",")
df_train["fallo_bin"] = df_train["fault_present"]
X_train = df_train.drop(["fault_present", "fallo_bin"], axis=1)

scaler = StandardScaler()
scaler.fit(X_train)

# 3. Crear servicio BentoML (sin runners)
svc = bentoml.Service("tep_service")

# 4. Crear app FastAPI
app = FastAPI()

# 5. Esquema de entrada
class InputData(BaseModel):
    features: list

# 6. Endpoint FastAPI (modelo directo)
@app.post("/predict")
def predict(input_data: InputData):

    X = np.array([input_data.features], dtype=float)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    return {
        "prediction": int(pred[0]),
        "probabilities": proba[0].tolist()
    }

# 7. Montar FastAPI dentro del servicio BentoML
svc.mount_asgi_app(app, "/")
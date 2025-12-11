import bentoml
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar modelo
model = bentoml.sklearn.load_model("tep_alerta_temprana:latest")

# Reconstruir scaler
df_train = pd.read_csv("DatasetProcesado/TEP_features_train.csv")
HORIZON = 10
df_train['fallo_futuro'] = df_train['fault_present'].shift(-HORIZON).fillna(0).astype(int)
features_to_drop = ['fault_present', 'fallo_futuro', 'time_since_fault', 'fault_stage']
X_train = df_train.drop(columns=[col for col in features_to_drop if col in df_train.columns])

scaler = StandardScaler()
scaler.fit(X_train)

svc = bentoml.Service("tep_alerta_service")

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(input_data: InputData):
    X = np.array([input_data.features], dtype=float)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    return {
        "prediction": int(pred[0]),
        "probability_fallo": proba[0][1]
    }

svc.mount_asgi_app(app, "/")
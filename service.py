import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
import numpy as np

# Cargar modelo
model_ref = bentoml.sklearn.get("tep_random_forest:latest")
model_runner = model_ref.to_runner()

# Definir servicio BentoML
svc = bentoml.Service("tep_service", runners=[model_runner])

# Definir esquema de entrada
class InputData(BaseModel):
    features: list  # lista de floats con las características de la simulación

@svc.api(input=JSON(), output=JSON())
async def predict(input_data: InputData):
    # Transformar datos con el scaler guardado
    scaler = model_runner.custom_objects["scaler"]
    X = scaler.transform([input_data.features])
    
    pred = model_runner.predict.run(X)
    proba = model_runner.predict_proba.run(X).tolist()[0]
    
    return {"prediction": int(pred[0]), "probabilities": proba}

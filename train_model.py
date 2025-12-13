#Crea modelos solo con top-10 features para modelo 1 y 2.
#Guarda los pickles en Modelos/.
#Guarda los rankings top-10 en DatasetProcesado/.
#Registra ambos modelos en BentoML con metadata de top-10 features.
#Evita reentrenar si los pickles ya existen.


import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import bentoml

# =========================
# Función para entrenar un modelo con top-N features
# =========================
def entrenar_modelo_topN(model_number=1, top_n=10):
    """
    Entrena un modelo RandomForest solo con las top-N features.
    Retorna: clf, ranking_top, metrics
    """
    # --- Cargar dataset ---
    df = pd.read_csv("DatasetProcesado/TEP_features_train.csv")

    # --- Configuración por modelo ---
    if model_number == 1:
        target_col = 'fault_present'
        model_pickle = "Modelos/model1_rf_top10.pkl"
        ranking_file = "DatasetProcesado/feature_importance_ranking_model1_top10.csv"
    else:
        target_col = 'fault_stage'
        model_pickle = "Modelos/model2_rf_top10.pkl"
        ranking_file = "DatasetProcesado/feature_importance_ranking_model2_top10.csv"

    non_feature_cols = ['faultNumber','simulationRun','sample','fault_present','time_since_fault','fault_stage']
    feature_columns = [col for col in df.columns if col not in non_feature_cols]

    X = df[feature_columns]
    y = df[target_col]

    # --- Escalado ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Random Forest temporal para ranking ---
    clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_temp.fit(X_scaled, y)

    # --- Ranking completo ---
    ranking = pd.DataFrame({'feature': X.columns, 'importance': clf_temp.feature_importances_})
    ranking_sorted = ranking.sort_values(by='importance', ascending=False)

    # --- Seleccionar top-N features ---
    top_features = ranking_sorted['feature'].head(top_n).tolist()
    ranking_top = ranking_sorted.head(top_n)

    # --- Guardar ranking top-N ---
    os.makedirs(os.path.dirname(ranking_file), exist_ok=True)
    ranking_top.to_csv(ranking_file, index=False)

    # --- Preparar X final con top-N ---
    X_final = df[top_features]
    X_final_scaled = scaler.fit_transform(X_final)

    # --- Entrenamiento final ---
    clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_final.fit(X_final_scaled, y)

    # --- Métricas ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_final_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = clf_final.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'features_used': top_features
    }

    # --- Guardar modelo pickle ---
    os.makedirs(os.path.dirname(model_pickle), exist_ok=True)
    joblib.dump(clf_final, model_pickle)

    print(f"Modelo {model_pickle} entrenado y guardado con top-{top_n} features.")

    return clf_final, ranking_top, metrics

# =========================
# Función para registrar en BentoML
# =========================
def save_model_bentoml(model_number=1):
    """
    Guarda un modelo entrenado en BentoML usando pickle existente.
    """
    if model_number == 1:
        model_pickle = "Modelos/model1_rf_top10.pkl"
        ranking_file = "DatasetProcesado/feature_importance_ranking_model1_top10.csv"
        model_name = "tep_model1"
        dataset_name = "TEP binary fault"
    else:
        model_pickle = "Modelos/model2_rf_top10.pkl"
        ranking_file = "DatasetProcesado/feature_importance_ranking_model2_top10.csv"
        model_name = "tep_model2"
        dataset_name = "TEP horizon fault"

    # --- Entrenar si no existe pickle ---
    if not os.path.exists(model_pickle):
        print(f"{model_pickle} no encontrado, entrenando modelo {model_number}...")
        entrenar_modelo_topN(model_number, top_n=10)

    # --- Cargar modelo y top-10 features ---
    clf = joblib.load(model_pickle)
    ranking_top = pd.read_csv(ranking_file)
    top_features = ranking_top['feature'].tolist()

    # --- Guardar en BentoML ---
    bento_model = bentoml.sklearn.save_model(
        model_name,
        clf,
        metadata={
            "top_features": top_features,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "scikit-learn",
            "dataset": dataset_name
        }
    )
    print(f"Modelo {model_name} guardado en BentoML store: {bento_model}")

# =========================
# Script principal
# =========================
if __name__ == "__main__":
    save_model_bentoml(1)  # Modelo 1
    save_model_bentoml(2)  # Modelo 2
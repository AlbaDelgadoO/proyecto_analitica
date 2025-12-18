import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest

import json

@st.cache_data
def load_top_features():
    with open("DatasetProcesado/top_features.json", "r") as f:
        return json.load(f)

TOP_FEATURES = load_top_features()


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="TEP - Detección de Fallos",
    page_icon="🏭",
    layout="wide"
)

# --- 1. FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data
def load_data():
    data_dir = "DatasetReducido" 
    try:
        df_normal = pd.read_csv(f"{data_dir}/FaultFree_Training_reduced.csv")
        df_faulty = pd.read_csv(f"{data_dir}/Faulty_Training_reduced.csv")
        
        # Etiquetas y limpieza básica
        df_normal['fault_present'] = 0
        df_normal['faultNumber'] = 0
        df_faulty['fault_present'] = 1
        
        # Concatenar
        df_combined = pd.concat([df_normal, df_faulty], ignore_index=True)
        return df_combined
    except FileNotFoundError:
        return None

# Definición de variables
PROCESS_VARS = [f"xmeas_{i}" for i in range(1, 42)]
ACTUATOR_VARS = [f"xmv_{i}" for i in range(1, 12)]
ALL_SENSORS = PROCESS_VARS + ACTUATOR_VARS

# Carga inicial
df = load_data()

# BARRA LATERAL (NAVEGACIÓN)
st.sidebar.image("img/menu.png", width=50)
page = st.sidebar.radio("Ir a:", ["Análisis Exploratorio (EDA)", "Ingeniería de Características", "Modelado y Entrenamiento", "Predicción vía API BentoML"])

if df is None:
    st.error("No se encontraron los archivos CSV.")
    st.stop()


# PÁGINA: ANÁLISIS EXPLORATORIO (EDA)
if page == "Análisis Exploratorio (EDA)":
    st.image("img/analisis.png", width=100)
    st.title("Análisis Exploratorio de Datos")
    st.markdown("Exploración interactiva de los sensores y la distribución de fallos.")

    # Definición de Colores
    MAIN_COLOR = "#2877FF"  
    SECONDARY_COLOR = "#25ED21"
    # Pestañas internas para organizar el EDA
    tab1, tab2, tab3, tab4 = st.tabs(["Resumen y Target", "Distribuciones Univariables", "Series Temporales", "PCA & Correlaciones"])

    with tab1:
        # FILA 1: TABLA 
        st.subheader("Estadísticas Descriptivas")
        st.markdown("Resumen estadístico de las primeras variables.")
        st.dataframe(df[ALL_SENSORS].describe().T.head(10), use_container_width=True)
        
        st.divider() 
        
        # FILA 2: GRÁFICOS 
        col_graph1, spacer, col_graph2 = st.columns([2, 0.2, 1.5]) 
        
        with col_graph1:
            st.subheader("Distribución de Fallos")
            
            fig_count, ax_count = plt.subplots(figsize=(8, 5)) 
            
            unique_faults = sorted(df['faultNumber'].unique())
            custom_palette = [SECONDARY_COLOR if x == 0 else MAIN_COLOR for x in unique_faults]
            
            sns.countplot(data=df, x='faultNumber', palette=custom_palette, ax=ax_count)
            
            ax_count.set_title('Muestras por Tipo de Fallo (0 = Normal)', fontsize=10)
            ax_count.set_ylabel('Cantidad')
            ax_count.set_xlabel('Tipo de Fallo')
            ax_count.tick_params(axis='x', rotation=0, labelsize=8)
            
            sns.despine(ax=ax_count)
            st.pyplot(fig_count)

        with col_graph2:
            st.subheader("Balance Global")
            
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            
            counts = df['fault_present'].value_counts()
            
            if counts.index[0] == 1:
                labels = ['Fallo (1)', 'Normal (0)']
                colors = [MAIN_COLOR, SECONDARY_COLOR] 
            else:
                labels = ['Normal (0)', 'Fallo (1)']
                colors = [SECONDARY_COLOR, MAIN_COLOR]
            
            ax_pie.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, 
                       startangle=90, textprops={'fontsize': 10})
            
            ax_pie.set_title('Normal vs. Fallo', fontsize=10)
            st.pyplot(fig_pie)

    with tab2:
        st.subheader("Comportamiento de Sensores y Actuadores")
        st.markdown("Comparativa de distribuciones. Selecciona múltiples variables para verlas en cuadrícula.")

        # 2. Creamos sub-pestañas 
        tab_act, tab_proc = st.tabs(["Variables Manipuladas (XMV)", "Variables de Proceso (XMEAS)"])

        # Función Helper
        def plot_grid(variable_list, default_selection):
            selected_vars = st.multiselect(
                "Selecciona variables para visualizar:", 
                options=variable_list, 
                default=default_selection
            )
            
            if not selected_vars:
                st.warning("Selecciona al menos una variable.")
                return

            cols = st.columns(2)
            
            for i, var in enumerate(selected_vars):
                col = cols[i % 2]
                with col:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    sns.kdeplot(
                        data=df, x=var, hue='fault_present', fill=True, common_norm=False, 
                        palette={0: SECONDARY_COLOR, 1: MAIN_COLOR},
                        alpha=0.3, linewidth=2, ax=ax
                    )
                    
                    ax.set_title(f"{var}", fontsize=10, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel('Densidad', fontsize=8)
                    
                    ax.legend(labels=['Fallo', 'Normal'], fontsize=8)
                    
                    sns.despine(ax=ax)
                    st.pyplot(fig)

        # PESTAÑA 1: ACTUADORES (XMV) 
        with tab_act:
            st.caption("Estas variables son **acciones de control** (Apertura de válvulas, Velocidades).")
            plot_grid(ACTUATOR_VARS, default_selection=ACTUATOR_VARS)

        # PESTAÑA 2: PROCESO (XMEAS) 
        with tab_proc:
            st.caption("Estas variables son **lecturas de sensores** (Temperaturas, Presiones, Niveles).")
            plot_grid(PROCESS_VARS, default_selection=['xmeas_1', 'xmeas_9', 'xmeas_21', 'xmeas_10'])

    with tab3:
        st.subheader("Dinámica Temporal")
        col_run1, col_run2 = st.columns(2)
        
        # Selectores de Runs
        runs_normal = df[df['fault_present'] == 0]['simulationRun'].unique()
        runs_faulty = df[df['fault_present'] == 1]['simulationRun'].unique()
        
        run_n = col_run1.selectbox("Run Normal:", runs_normal)
        run_f = col_run2.selectbox("Run con Fallo:", runs_faulty)
        var_temp = st.selectbox("Variable a visualizar en el tiempo:", ALL_SENSORS, index=8) 

        # Filtrado
        sim_normal = df[(df['simulationRun'] == run_n) & (df['fault_present'] == 0)]
        sim_faulty = df[(df['simulationRun'] == run_f) & (df['fault_present'] == 1)]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sim_normal['sample'], sim_normal[var_temp], label=f'Normal (Run {run_n})', color=SECONDARY_COLOR)
        ax.plot(sim_faulty['sample'], sim_faulty[var_temp], label=f'Fallo (Run {run_f})', color=MAIN_COLOR, alpha=0.8)
        ax.legend()
        ax.set_xlabel("Tiempo (Muestra)")
        ax.set_ylabel(var_temp)
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with tab4:
        st.subheader("Reducción de Dimensionalidad (PCA)")
        st.markdown("Proyección de las 52 variables en 2 componentes principales.")

        if st.button("Calcular y Visualizar PCA (Puede tardar unos segundos)"):
            with st.spinner("Calculando PCA..."):
                # 1. Muestreo y Preparación
                df_sample = df.sample(min(10000, len(df)), random_state=42)
                X = df_sample[ALL_SENSORS]
                y = df_sample['fault_present']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 2. Cálculo PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # 3. Visualización con SEABORN (Para usar tu paleta)
                fig, ax = plt.subplots(figsize=(8, 6))
                
                sns.scatterplot(
                    x=X_pca[:, 0], 
                    y=X_pca[:, 1], 
                    hue=y, 
                    palette={0: SECONDARY_COLOR, 1: MAIN_COLOR}, 
                    alpha=0.6, 
                    s=20, 
                    edgecolor=None, 
                    ax=ax
                )
                
                ax.set_xlabel('Componente Principal 1')
                ax.set_ylabel('Componente Principal 2')
                ax.set_title(f'Varianza Explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%')
                
                # Ajustamos la leyenda manualmente para que sea legible
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, ['Normal', 'Fallo'], title="Estado")
                
                st.pyplot(fig)





# PÁGINA: INGENIERÍA DE CARACTERÍSTICAS
elif page == "Ingeniería de Características":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    st.image("img/data.png", width=100)
    st.title("Ingeniería de Características")
    st.markdown("""
    Esta sección sirve para ver las transformaciones aplicadas en el notebook de Ingeniería de Características.
    """)

    # Cargar dataset procesado
    try:
        df_feat = pd.read_csv("DatasetProcesado/TEP_features_train.csv")
    except Exception as e:
        st.error(f"No se pudo cargar el dataset procesado: {e}")
        st.stop()

    st.subheader("Vista general del dataset procesado")
    st.write(f"Shape: **{df_feat.shape[0]} filas × {df_feat.shape[1]} columnas**")
    st.dataframe(df_feat.head(), use_container_width=True)
    st.divider()

    # Selección interactiva de simulación y variable
    st.subheader("Comparación de transformaciones")
    run = st.selectbox("Simulation Run", sorted(df_feat["simulationRun"].unique()))
    var = st.selectbox("Variable a analizar", [c for c in df_feat.columns if c.startswith("xmeas_")])

    df_run = df_feat[df_feat["simulationRun"] == run].sort_values("sample")

    # Gráfico interactivo: Original vs Transformaciones temporales
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_run["sample"], df_run[var], label="Original", alpha=0.8)
    
    # Agregar medias móviles y std
    for w in [5, 10]:
        ma_col = f"{var}_ma{w}"
        std_col = f"{var}_std{w}"
        if ma_col in df_run.columns:
            ax.plot(df_run["sample"], df_run[ma_col], label=f"MA {w}")
        if std_col in df_run.columns:
            ax.plot(df_run["sample"], df_run[std_col], label=f"STD {w}")

    # Inicio del fallo
    if df_run["fault_present"].max() == 1:
        x0 = df_run[df_run["fault_present"] == 1]["sample"].min()
        ax.axvline(x=x0, linestyle="--", color="red", label="Inicio fallo")

    ax.set_title(f"Simulación {run} - Variable {var}")
    ax.set_xlabel("Sample")
    ax.legend()
    st.pyplot(fig)
    st.divider()

    # Distribución de variable escalada
    st.subheader("Distribución de variable escalada (Normal vs Fallo)")
    var_scaled = f"{var}_scaled"
    if var_scaled in df_feat.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.kdeplot(df_feat[df_feat.fault_present == 0][var_scaled], fill=True, label="Normal", ax=ax2)
        sns.kdeplot(df_feat[df_feat.fault_present == 1][var_scaled], fill=True, label="Fallo", ax=ax2)
        ax2.set_title(f"Distribución de {var_scaled}")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning(f"No existe la feature escalada: {var_scaled}")
    st.divider()

    # Características más importantes 
    st.subheader("Top 20 Features más importantes")

    ranking_path = "modelos/feature_importance_ranking_model1.csv"
    if os.path.exists(ranking_path):
        ranking = pd.read_csv(ranking_path)
        st.dataframe(ranking.head(20), use_container_width=True)
    else:
        st.info("No se encontró un ranking de importancia guardado.")




# PÁGINA: MODELADO Y ENTRENAMIENTO
elif page == "Modelado y Entrenamiento":

    st.image("img/entrenamiento.png", width=100)
    st.title("Laboratorio de Modelos ML")
    st.markdown("Entrena y evalúa modelos de clasificación usando selección por importancia.")

    # SELECCIÓN DE MODELO
    modelo_sel = st.selectbox(
        "Selecciona el modelo:",
        [
            "Clasificación Binaria (Fallo vs Normal)",
            "Predicción de Fallos Futuros (Horizonte)",
            "Clasificación Multiclase del Tipo de Fallo",
            "Detección de anomalías con Isolation Forest",
            "Detección de anomalías con Autoencoder"
        ]
    )

    unsupervised_models = [
        "Detección de anomalías con Isolation Forest",
        "Detección de anomalías con Autoencoder"
    ]

    is_unsupervised = modelo_sel in unsupervised_models

    if modelo_sel == "Clasificación Binaria (Fallo vs Normal)":
        target_col   = "fault_present"
        ranking_base = "modelos/feature_importance_ranking_model1.csv"
        model_base   = "modelos/model1_rf"
        
    elif modelo_sel == "Predicción de Fallos Futuros (Horizonte)":
        target_col   = "fault_stage"
        ranking_base = "modelos/feature_importance_ranking_model2.csv"
        model_base   = "modelos/model2_rf"
    
    else:  # MODELO 3
        target_col   = "faultNumber"
        ranking_base = "modelos/feature_importance_ranking_model3.csv"
        model_base   = "modelos/model3_rf"

    os.makedirs("DatasetProcesado", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    if not is_unsupervised:
        # CARGA DEL RANKING BASE
        df_cols = pd.read_csv(
            "DatasetProcesado/TEP_features_train.csv",
            nrows=1
        ).columns.tolist()

        if os.path.exists(ranking_base):
            ranking_full = pd.read_csv(ranking_base)
            all_features = [f for f in ranking_full["feature"] if f in df_cols]
        else:
            st.error(f"No existe el ranking base: {ranking_base}")
            st.stop()

        # SLIDER TOP-N FEATURES
        n_features = st.slider(
            "Número de features a usar",
            min_value=5,
            max_value=len(all_features),
            value=min(20, len(all_features)),
            step=1,
            key=f"slider_{modelo_sel}"
        )

        top_features = all_features[:n_features]
        st.write(f"Se entrenará con **{n_features} features**")

        # Nombres finales
        model_path   = f"{model_base}_top{n_features}.pkl"
        ranking_path = f"{ranking_base.replace('.csv', f'_top{n_features}.csv')}"

        # ENTRENAMIENTO
        if st.button("Entrenar modelo"):

            cols_to_load = top_features + [target_col]

            df = pd.read_csv(
                "DatasetProcesado/TEP_features_train.csv",
                usecols=cols_to_load
            )

            X = df[top_features].values
            y = df[target_col].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # MODELO OPTIMIZADO PARA STREAMLIT
            clf = RandomForestClassifier(
                n_estimators=40,
                max_depth=12,
                min_samples_leaf=30,
                random_state=42,
                n_jobs=-1
            )

            with st.spinner("Entrenando modelo..."):
                clf.fit(X_train, y_train)

            # GUARDADO
            ranking_top = pd.DataFrame({
                "feature": top_features,
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False)

            ranking_top.to_csv(ranking_path, index=False)

            joblib.dump(
                {
                    "model": clf,
                    "scaler": scaler,
                    "features": top_features
                },
                model_path
            )

            st.success(f"Modelo guardado: {model_path}")
            st.success(f"Ranking guardado: {ranking_path}")

            # EVALUACIÓN
            y_pred = clf.predict(X_test)

            st.subheader("Evaluación del Modelo")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            st.write(f"**F1-score:** {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Valor real")
            st.pyplot(fig)

    # Unsupervised Models
    if modelo_sel == "Detección de anomalías con Isolation Forest":

        st.subheader("Detección de anomalías con Isolation Forest")

        model_path = "modelos/isolation_forest.pkl"

        if not os.path.exists(model_path):
            st.error("No se encontró el modelo Isolation Forest entrenado.")
            st.stop()

        # Cargar modelo
        artifact = joblib.load(model_path)
        iso     = artifact["model"]
        scaler  = artifact["scaler"]
        features = artifact["features"]

        # Cargar datos de test
        df_test = pd.read_csv("DatasetProcesado/TEP_features_test.csv")

        X_test = df_test[features]
        y_true = df_test["fault_present"]

        X_test_scaled = scaler.transform(X_test)

        # Predicción
        y_pred = iso.predict(X_test_scaled)
        y_pred = np.where(y_pred == -1, 1, 0)  # 1 = anomalía

        # Métricas
        st.subheader("Resultados")

        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        st.text("Classification Report")
        st.text(classification_report(y_true, y_pred))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Reds", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        st.pyplot(fig)

        # ROC-AUC
        scores = -iso.decision_function(X_test_scaled)
        auc = roc_auc_score(y_true, scores)
        st.write(f"**ROC-AUC:** {auc:.4f}")

        # Score temporal
        st.subheader("Puntuación de anomalía")

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(scores)
        ax2.set_xlabel("Muestra")
        ax2.set_ylabel("Anomalía")
        st.pyplot(fig2)
    
    elif modelo_sel == "Detección de anomalías con Autoencoder":

        st.subheader("Detección de anomalías con Autoencoder")

        model_path = "modelos/autoencoder_model.pkl"

        if not os.path.exists(model_path):
            st.error("No se encontraron los archivos del Autoencoder entrenado.")
            st.stop()

        # Cargar modelo y metadata
        artifact = joblib.load(model_path)
        autoencoder = artifact["model"]
        scaler      = artifact["scaler"]
        threshold   = artifact["threshold"]
        features    = artifact["features"]


        # Datos de test
        df_test = pd.read_csv("DatasetProcesado/TEP_features_test.csv")

        X_test = df_test[features].fillna(0).values
        y_true = df_test["fault_present"].values

        X_test_scaled = scaler.transform(X_test)

        # Reconstrucción
        X_recon = autoencoder.predict(X_test_scaled, verbose=0)

        mse = np.mean(np.square(X_test_scaled - X_recon), axis=1)

        y_pred = (mse > threshold).astype(int)

        # Métricas
        st.subheader("Resultados")

        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        st.text("Classification Report")
        st.text(classification_report(y_true, y_pred))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Oranges", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        st.pyplot(fig)

        auc = roc_auc_score(y_true, mse)
        st.write(f"**ROC-AUC:** {auc:.4f}")

        # Error de reconstrucción
        st.subheader("Error de reconstrucción")

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(mse)
        ax2.axhline(threshold, linestyle="--", label="Threshold")
        ax2.legend()
        st.pyplot(fig2)

# ==========================================
# PÁGINA 3: PREDICCIÓN VÍA API BENTOML (CORREGIDA FINAL)
# ==========================================
elif page == "Predicción vía API BentoML":
    st.image("img/modelo.png", width=100)
    st.title("Panel de Inferencia en Tiempo Real")
    st.markdown("Diagnóstico industrial mediante modelos servidos en BentoML.")

    # 1. Carga de datos de test
    try:
        df_test = pd.read_csv("DatasetProcesado/TEP_features_test.csv")
    except Exception as e:
        st.error(f"No se encontró el dataset de test: {e}")
        st.stop()

    st.subheader("1. Selección de Datos de Entrada")
    
    if 'idx_test' not in st.session_state:
        st.session_state.idx_test = 0
            
    col_idx, col_btn = st.columns([2, 1])

    idx = st.number_input("Índice de fila:", 1, len(df_test)-1, value=max(1, st.session_state.idx_test))
    st.session_state.idx_test = idx

    if st.button("Fila aleatoria"):
        st.session_state.idx_test = np.random.randint(0, len(df_test))
        st.rerun()
            
    fila_raw = df_test.iloc[[st.session_state.idx_test]]
    
    st.write(f"Muestra de la fila seleccionada: **{st.session_state.idx_test}**")
    st.dataframe(fila_raw.iloc[:, :12], use_container_width=True)

    st.divider()

    st.subheader("2. Ejecución de Modelos")
    modelo_api = st.selectbox("Selecciona el modelo para la inferencia:", [
        "Binario (Normal vs Fallo)", 
        "Horizonte (Detección Temprana)",
        "Multiclase (Tipo de Fallo)", 
        "Isolation Forest (Anomalías)",
        "Autoencoder (Reconstrucción)"
    ])

    endpoints = {
        "Binario (Normal vs Fallo)": "predict_binary",
        "Horizonte (Detección Temprana)": "predict_horizon",
        "Multiclase (Tipo de Fallo)": "predict_multiclass",
        "Isolation Forest (Anomalías)": "predict_isolation",
        "Autoencoder (Reconstrucción)": "predict_autoencoder"
    }

    if st.button("Ejecutar Diagnóstico"):
        cols_to_drop = ["faultNumber", "fault_present", "simulationRun", "sample"]
        
        # SELECCIÓN CORRECTA DE FEATURES (CLAVE DEL PROBLEMA)
        

        # 1. Quitamos columnas no numéricas / labels
        df_api = fila_raw.drop(columns=[c for c in cols_to_drop if c in fila_raw.columns])

        # 2. Verificación dura
        missing = [f for f in TOP_FEATURES if f not in df_api.columns]
        extra   = [c for c in df_api.columns if c not in TOP_FEATURES]

        if missing:
            st.error(f"Faltan features requeridas por el modelo: {missing}")
            st.stop()

        # 3. Selección y ORDEN EXACTO
        df_api = df_api[TOP_FEATURES]

        # 4. Limpieza
        df_api = df_api.replace([np.inf, -np.inf], np.nan)
        df_api = df_api.fillna(0)

        # 5. Conversión final
        input_data = df_api.values.astype(np.float32)

        try:
            with st.spinner(f"Consultando {modelo_api}..."):
                url = f"http://localhost:3000/{endpoints[modelo_api]}"
                response = requests.post(url, json=input_data.tolist())
                
                if response.status_code == 200:
                    res = response.json()
                    st.success("¡Diagnóstico recibido correctamente!")
                    
                    pred = res.get("prediction")
                    
                    # Dashboard de métricas
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if "Multiclase" in modelo_api:
                            label = "✅ NORMAL" if pred == 0 else f"🚨 FALLO TIPO {pred}"
                        else:
                            label = "🚨 FALLO" if pred == 1 else "✅ NORMAL"
                        st.metric("Resultado", label)
                    
                    with c2:
                        if "mse" in res:
                            st.metric("Error Reconstrucción", f"{res['mse']:.4f}")
                        elif "Multiclase" in modelo_api:
                            st.metric("ID de Clase", pred)

                    with c3:
                        if "threshold" in res:
                            st.metric("Umbral Crítico", f"{res['threshold']:.4f}")

                    with st.expander("Ver detalles técnicos (JSON)"):
                        st.json(res)
                else:
                    st.error(f"Error de la API ({response.status_code}): {response.text}")
                    
        except Exception as e:
            st.error(f"Fallo de conexión con BentoML: {e}")
            st.info("Asegúrate de que BentoML está corriendo en la terminal con 'bentoml serve'")
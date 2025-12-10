import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
page = st.sidebar.radio("Ir a:", ["Análisis Exploratorio (EDA)", "Modelado y Entrenamiento"])

if df is None:
    st.error("No se encontraron los archivos CSV.")
    st.stop()

# ==========================================
# PÁGINA 1: ANÁLISIS EXPLORATORIO (EDA)
# ==========================================
if page == "Análisis Exploratorio (EDA)":
    st.image("img/data.png", width=100)
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

# # ==========================================
# # PÁGINA 2: MODELADO Y ENTRENAMIENTO 
# # ==========================================
# elif page == "Modelado y Entrenamiento":
#     st.title("Laboratorio de Modelos ML")
#     st.markdown("Entrena y evalúa modelos de clasificación en tiempo real.")

#     # --- Selección de Modelo ---
#     modelo_sel = st.selectbox(
#         "Selecciona el modelo que deseas entrenar:",
#         ["Clasificación Binaria (Fallo vs Normal)", 
#          "Predicción de Fallos Futuros (Horizonte)", 
#          "Clasificación Multiclase de Tipo de Fallo"]
#     )

#     # --- Selección de Variables ---
#     variables_sel = st.multiselect(
#         "Selecciona variables predictoras:", 
#         options=ALL_SENSORS,
#         default=ALL_SENSORS
#     )

#     if not variables_sel:
#         st.warning("Debes seleccionar al menos una variable.")
#         st.stop()

#     # --- Botón para Entrenar ---
#     if st.button("Entrenar Modelo"):
#         st.info("Entrenando modelo... Esto puede tardar unos segundos.")
        
#         # Preparación de datos
#         if modelo_sel == "Clasificación Binaria (Fallo vs Normal)":
#             target = "fault_present"
#             X = df[variables_sel]
#             y = df[target]
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
#             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#             clf = RandomForestClassifier(n_estimators=100, random_state=42)
#             clf.fit(X_train, y_train)

#         elif modelo_sel == "Predicción de Fallos Futuros (Horizonte)":
#             target = "fault_present"  # Ajusta si tienes otra variable para el horizonte
#             X = df[variables_sel]
#             y = df[target]
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
#             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#             clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
#             clf.fit(X_train, y_train)

#         elif modelo_sel == "Clasificación Multiclase de Tipo de Fallo":
#             target = "faultNumber"
#             X = df[variables_sel]
#             y = df[target]
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#             clf.fit(X_train, y_train)

#         # --- Evaluación ---
#         st.subheader("Evaluación del Modelo")
#         y_pred = clf.predict(X_test)

#         st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
#         st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.2f}")
#         st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.2f}")
#         st.write(f"**F1-score:** {f1_score(y_test, y_pred, average='weighted'):.2f}")

#         st.subheader("Matriz de Confusión")
#         fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
#         sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
#         ax_cm.set_xlabel("Predicción")
#         ax_cm.set_ylabel("Valor Real")
#         st.pyplot(fig_cm)

#         # --- Predicción en Tiempo Real ---
#         st.subheader("Predicción en Tiempo Real")
#         uploaded_file = st.file_uploader("Sube archivo CSV con nuevas simulaciones", type=["csv"])
#         if uploaded_file:
#             df_new = pd.read_csv(uploaded_file)
#             if set(variables_sel).issubset(df_new.columns):
#                 X_new = df_new[variables_sel]
#                 # Escalado si aplica
#                 if modelo_sel != "Clasificación Multiclase de Tipo de Fallo":
#                     X_new = scaler.transform(X_new)
#                 preds = clf.predict(X_new)
#                 df_new["Predicción"] = preds
#                 st.dataframe(df_new.head(20), use_container_width=True)
#             else:
#                 st.error("El CSV no contiene todas las variables seleccionadas.")


# ==========================================
# PÁGINA 2: MODELADO Y ENTRENAMIENTO / PREDICCIÓN BENTO
# ==========================================
elif page == "Modelado y Entrenamiento":
    st.title("Laboratorio de Modelos ML")
    st.markdown("Entrena y evalúa modelos de clasificación en tiempo real.")

    # --- Selección de Modelo ---
    modelo_sel = st.selectbox(
        "Selecciona el modelo que deseas entrenar:",
        ["Clasificación Binaria (Fallo vs Normal)", 
         "Predicción de Fallos Futuros (Horizonte)", 
         "Clasificación Multiclase de Tipo de Fallo"]
    )

    # --- Selección de Variables ---
    variables_sel = st.multiselect(
        "Selecciona variables predictoras:", 
        options=ALL_SENSORS,
        default=ALL_SENSORS
    )

    if not variables_sel:
        st.warning("Debes seleccionar al menos una variable.")
        st.stop()

    # --- Botón para Entrenar ---
    if st.button("Entrenar Modelo Local"):
        st.info("Entrenando modelo... Esto puede tardar unos segundos.")
        
        # Preparación de datos y entrenamiento
        if modelo_sel == "Clasificación Binaria (Fallo vs Normal)":
            target = "fault_present"
            X = df[variables_sel]
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

        elif modelo_sel == "Predicción de Fallos Futuros (Horizonte)":
            target = "fault_present"  # Ajusta si tienes otra variable para el horizonte
            X = df[variables_sel]
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)

        elif modelo_sel == "Clasificación Multiclase de Tipo de Fallo":
            target = "faultNumber"
            X = df[variables_sel]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

        # --- Evaluación del Modelo Local ---
        st.subheader("Evaluación del Modelo Local")
        y_pred = clf.predict(X_test)

        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"**F1-score:** {f1_score(y_test, y_pred, average='weighted'):.2f}")

        st.subheader("Matriz de Confusión")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Valor Real")
        st.pyplot(fig_cm)

        # --- Predicción en Tiempo Real Local ---
        st.subheader("Predicción en Tiempo Real (Local)")
        uploaded_file = st.file_uploader("Sube archivo CSV con nuevas simulaciones", type=["csv"])
        if uploaded_file:
            df_new = pd.read_csv(uploaded_file)
            if set(variables_sel).issubset(df_new.columns):
                X_new = df_new[variables_sel]
                if modelo_sel != "Clasificación Multiclase de Tipo de Fallo":
                    X_new = scaler.transform(X_new)
                preds = clf.predict(X_new)
                df_new["Predicción"] = preds
                st.dataframe(df_new.head(20), use_container_width=True)
            else:
                st.error("El CSV no contiene todas las variables seleccionadas.")

# ==========================================
# NUEVA PÁGINA: PREDICCIÓN VÍA API BENTO
# ==========================================
elif page == "Predicción vía API BentoML":
    st.title("Predicción en Tiempo Real usando BentoML API")
    st.markdown("Introduce los valores de las 52 variables y obtén la predicción del modelo servido por BentoML.")

    # Crear sliders para las 52 variables
    features = [st.slider(f"{var}", 0.0, 10.0, 0.5) for var in ALL_SENSORS]

    if st.button("Predecir con API"):
        try:
            response = requests.post(
                "http://localhost:3000/predict",
                json={"features": features}
            )
            if response.status_code == 200:
                result = response.json()
                st.write("Predicción:", result["prediction"])
                st.write("Probabilidades:", result["probabilities"])
            else:
                st.error(f"Error al conectar con API: {response.status_code}")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")


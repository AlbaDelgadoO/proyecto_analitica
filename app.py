import streamlit as st
import pyreadr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# Configuración de página de Streamlit
st.set_page_config(
    page_title="TEP Fault Detection Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------
# Función para cargar RData con Caching
# --------------------------------------
@st.cache_data
def load_rdata(file_path: str) -> pd.DataFrame:
    """
    Lee un archivo RData usando pyreadr y devuelve el primer DataFrame contenido.
    Usa el decorador st.cache_data para evitar recargar los datos en cada interacción.
    """
    try:
        # st.spinner() muestra un mensaje de carga mientras se ejecuta la función
        with st.spinner(f'Cargando {file_path}...'):
            result = pyreadr.read_r(file_path)
        
        # Obtener el nombre de la clave y devolver el DataFrame
        df_name = list(result.keys())[0]
        st.info(f"Archivo {file_path} cargado con el nombre de objeto '{df_name}'.")
        return result[df_name]
    
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la misma carpeta que 'app.py'.")
        return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error
    except Exception as e:
        st.error(f"Error al leer '{file_path}': {e}")
        return pd.DataFrame()

# --------------------------------------
# Cargar todos los datasets
# --------------------------------------
st.title('🏭 Tennessee Eastman Process (TEP) - Análisis de Fallos')
st.markdown("Herramienta interactiva para visualizar las series temporales de variables del proceso TEP, enfocada solo en datos con fallos.")

# Diccionario de archivos y DataFrames
datasets: Dict[str, pd.DataFrame] = {
    "Fault Free Training": load_rdata("TEP_FaultFree_Training.RData"),
    "Fault Free Testing": load_rdata("TEP_FaultFree_Testing.RData"),
    "Faulty Training": load_rdata("TEP_Faulty_Training.RData"),
    "Faulty Testing": load_rdata("TEP_Faulty_Testing.RData")
}

# --------------------------------------
# PRE-PROCESAMIENTO: Definir y etiquetar solo los conjuntos con Fallo
# --------------------------------------

# 1. Conjunto de Entrenamiento (Solo datos con fallo)
df_train_raw = datasets["Faulty Training"].copy()
# CREACIÓN DE LA COLUMNA CRÍTICA: fault_present indica la presencia del fallo (0:Normal, 1:Fallo)
df_train_raw['fault_present'] = np.where(df_train_raw['faultNumber'] > 0, 1, 0) 

# 2. Conjunto de Prueba (Solo datos con fallo)
df_test_raw = datasets["Faulty Testing"].copy()
df_test_raw['fault_present'] = np.where(df_test_raw['faultNumber'] > 0, 1, 0) 

# --------------------------------------
# Interfaz de Usuario en la Barra Lateral
# --------------------------------------
st.sidebar.header('Parámetros de Visualización')

# Selector para elegir el conjunto de datos (Entrenamiento o Prueba)
dataset_choice = st.sidebar.radio(
    "Selecciona el Conjunto de Datos:",
    ('Entrenamiento (Training)', 'Prueba (Testing)')
)

# Asignar el DataFrame de análisis basado en la selección
if dataset_choice == 'Entrenamiento (Training)':
    df_analisis = df_train_raw.copy()
else:
    df_analisis = df_test_raw.copy()

# Obtener todas las simulaciones que contienen un fallo (fault_present == 1)
# Usamos el DataFrame procesado que contiene 'fault_present'
if 'fault_present' in df_analisis.columns:
    faulty_runs = df_analisis[df_analisis['fault_present'] == 1]['simulationRun'].unique()
else:
    # Esto ocurre si el archivo 'Faulty Training' no se cargó correctamente
    faulty_runs = np.array([])
    
# Selector de la Simulación (Solo se muestran las que tienen fallo)
selected_run = st.sidebar.selectbox(
    '1. Simulación con Fallo (Run):',
    options=sorted(faulty_runs.tolist()) if faulty_runs.size > 0 else ['No Runs Disponibles'],
    help='Elige una simulación de la lista que contenga un fallo inyectado.'
)

# Obtener los nombres de las columnas 'xmeas_' y 'xmv_'
data_cols_xmeas = [col for col in df_analisis.columns if col.startswith('xmeas')]
data_cols_xmv = [col for col in df_analisis.columns if col.startswith('xmv')]
all_data_cols = data_cols_xmeas + data_cols_xmv

# Selector de la Variable
key_variable = st.sidebar.selectbox(
    '2. Variable de Proceso:',
    options=all_data_cols,
    index=all_data_cols.index('xmeas_1') if 'xmeas_1' in all_data_cols else 0
)

# --------------------------------------
# Gráfico de Series Temporales (Main Panel)
# --------------------------------------

st.header(f'📈 Serie Temporal de la Variable {key_variable} - {dataset_choice}')

if df_analisis.empty:
    st.error("El DataFrame de análisis está vacío. Revisa la carga de archivos.")
elif len(faulty_runs) == 0:
    st.warning("No se encontraron simulaciones con fallos (fault_present = 1) en el conjunto seleccionado. Asegúrate de que los archivos 'Faulty' se cargaron correctamente.")
else:
    # 1. Filtrar el DataFrame basado en la selección del usuario
    df_sim_fault = df_analisis[df_analisis['simulationRun'] == selected_run].copy()
    
    # 2. Identificar el punto de inicio del fallo (muestra más de 0) y el número de fallo
    fault_start_sample = df_sim_fault[df_sim_fault['fault_present'] == 1]['sample'].min()
    fault_number = df_sim_fault['faultNumber'].max() # Obtiene el número de fallo (1-20)

    # 3. Crear el gráfico interactivo con Plotly Express
    fig = px.line(
        df_sim_fault,
        x='sample',
        y=key_variable,
        # Usamos 'faultNumber' (convertido a string) para diferenciar los colores
        color=df_sim_fault['faultNumber'].astype(str), 
        title=f'Simulación {selected_run}: Fallo {int(fault_number)}',
        template="plotly_white"
    )

    # 4. Ajustes y Personalización de líneas y colores
    fig.update_traces(line=dict(width=2.5)) 

    # Renombrar las leyendas para mayor claridad
    newnames = {
        '0': 'Régimen Normal (No Fallo)',
        str(int(fault_number)): f'Fallo Activo ID {int(fault_number)}'
    }
    
    fig.for_each_trace(lambda t: t.update(
        name = newnames.get(t.name, t.name),
        legendgroup = newnames.get(t.name, t.name),
        hovertemplate = t.hovertemplate.replace(t.name, newnames.get(t.name, t.name))
    ))

    # 5. Añadir Sombreado para el Periodo de Fallo (variación)
    if pd.notna(fault_start_sample):
        # Añadir un rectángulo sombreado para el periodo de fallo
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y domain",
            x0=fault_start_sample,
            x1=df_sim_fault['sample'].max(), # hasta el final de la simulación
            y0=0, # Parte inferior del dominio Y
            y1=1, # Parte superior del dominio Y
            fillcolor="Red",
            opacity=0.15, # Opacidad para el sombreado
            layer="below",
            line_width=0,
        )
        
        # Añadir la línea vertical de inicio del fallo (más visible)
        fig.add_vline(
            x=fault_start_sample,
            line_dash="dot",
            line_color="darkred",
            line_width=2,
            annotation_text=" Inicio del Fallo ",
            annotation_position="top right"
        )

    # 6. Configuración final
    fig.update_layout(
        showlegend=True,
        legend_title_text='Estado del Proceso',
        xaxis_title='Muestra (Sample)',
        yaxis_title=key_variable,
        hovermode="x unified", # Muestra la información de hover de todas las líneas al mismo tiempo
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # 7. Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------
    # Vista de Datos de la Simulación
    # --------------------------------------
    st.subheader(f"Datos Crudos para Simulación {selected_run}")
    st.dataframe(df_sim_fault)
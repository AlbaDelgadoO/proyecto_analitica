# Proyecto Final – Analítica Industrial del Proceso Tennessee Eastman (TEP)

**Asignatura:** Analítica de Datos para la Industria  
**Curso:** 2025–2026   

## Integrantes del grupo
- Alba Delgado 
- Miren Lépée
- Oihane Camacho
- Elba Barrueta

## Descripción del proyecto
Este proyecto aplica técnicas de analítica de datos e inteligencia artificial al Proceso Tennessee Eastman (TEP), un benchmark industrial para la detección de fallos y optimización de procesos químicos.

Incluye:
- Reducción y procesamiento del dataset original.
- Análisis exploratorio y visualización de datos.
- Ingeniería de características para modelado.
- Entrenamiento y evaluación de modelos supervisados y no supervisados.
- Aplicación interactiva en Streamlit para exploración y predicción en tiempo real.
- Despliegue de API de inferencia con BentoML.

## Dataset utilizado
- **Nombre:** Tennessee Eastman Process (TEP)
- **Formato original:** `.RData`
- **Ficheros incluidos:**
  - `TEP_FaultFree_Training.RData`
  - `TEP_FaultFree_Testing.RData`
  - `TEP_Faulty_Training.RData`
  - `TEP_Faulty_Testing.RData`

Estos archivos se incluyen en la carpeta del proyecto. Sin embargo en caso de querer descargarlos, este es el enlace original.

**Enlace descarga dataset exacto utilizado:**  
[Harvard Dataverse – TEP Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)  

Los datasets originales se reducen y procesan mediante el notebook Reducir_Dataset, generando versiones optimizadas para el modelado.

## Estructura del proyecto
proyecto_analitica/
│
├── DatasetReducido/ # Generado tras ejecutar Reducir_Dataset.ipynb
├── DatasetProcesado/ # Generado tras ejecutar Ingenieria_de_caracteristicas.ipynb
├── img/ # Imágenes usadas en Streamlit
├── Modelos/ # Modelos entrenados (.pkl) y resultados
│
├── Reducir_Dataset.ipynb # Paso 1 – reducción del dataset
├── desc_dataset.ipynb # Paso 2 – análisis exploratorio
├── Ingenieria_de_caracteristicas.ipynb # Paso 3 – ingeniería de características
├── Modelos.ipynb # Paso 4 – entrenamiento y evaluación de modelos
│
├── app.py # Aplicación Streamlit
├── service.py # Servicio BentoML para inferencia
├── train_model.py # Script auxiliar de entrenamiento
│
├── bentoml/ # Artefactos generados por BentoML
│
├── TEP_FaultFree_Training.RData
├── TEP_FaultFree_Testing.RData
├── TEP_Faulty_Training.RData
├── TEP_Faulty_Testing.RData
└── README.md


## Flujo de ejecución
Este es el flujo de ejcución del trabajo completo. Se recomienda seguir el orden de ejecución de los archivos jupyter notebooks.

1. **Reducción del dataset**
   - Ejecutar `Reducir_Dataset.ipynb` para generar `DatasetReducido/`.

2. **Análisis exploratorio**
   - Ejecutar `desc_dataset.ipynb` para realizar EDA y justificación visual.

3. **Ingeniería de características**
   - Ejecutar `Ingenieria_de_caracteristicas.ipynb` para generar `DatasetProcesado/`.

4. **Entrenamiento y evaluación de modelos**
   - Ejecutar `Modelos.ipynb`:
     - **Modelo 1:** Clasificación binaria (fallo presente / no fallo).
     - **Modelo 2:** Predicción anticipada de fallos.
     - **Modelo 3:** Clasificación multiclase de fallos (21 clases).
     - **Modelos no supervisados:** Isolation Forest y Autoencoder para detección de anomalías.

5. **Aplicación Streamlit**
   - Ejecutar en la terminal, dentro de la carpeta del proyecto:
     ```bash
     streamlit run app.py
     ```
   - Permite realizar la exploración de datos, la visualización de métricas, el entrenamiento y la predicción en tiempo real.

6. **Despliegue con BentoML**
   - Ejecutar:
     ```bash
     bentoml serve service:svc
     ```
   - La API se integra con Streamlit para realizar predicciones en tiempo real.

## Entorno y dependencias
Para poder ejecutar los modelos 4 y 5 sera necesario realizar estos pasos:

**Inatalación recomendada:** Python 3.10 (64-bit) para Windows 
**Crear entorno virtual fuera de la carpeta del proyecto:**
Ajustar ruta al path con Python 3.10 instalado y ejecutar ```-m venv env_tep```
Para activar el entorno ```env_tep\Scripts\activate```
Comprobar versión ```python --version``` debe ser 3.10
Una vez dentro de env_tep se debe instalar lo siguiente:
```pip install numpy==1.26.4```
```pip install pandas scikit-learn joblib```
```pip install tensorflow==2.15.0```
```pip install streamlit seaborn matplotlib```

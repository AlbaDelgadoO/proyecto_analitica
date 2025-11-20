import pyreadr
import pandas as pd

# --------------------------------------
# Función para cargar RData
# --------------------------------------
def load_rdata(file_path):
    result = pyreadr.read_r(file_path)
    df_name = list(result.keys())[0]
    return result[df_name]

# --------------------------------------
# Cargar datasets
# --------------------------------------
datasets = {
    "Fault Free Training": load_rdata("TEP_FaultFree_Training.RData"),
    "Fault Free Testing": load_rdata("TEP_FaultFree_Testing.RData"),
    "Faulty Training": load_rdata("TEP_Faulty_Training.RData"),
    "Faulty Testing": load_rdata("TEP_Faulty_Testing.RData")
}

# --------------------------------------
# Variables de proceso y actuadores
# --------------------------------------
process_vars = [f"xmeas_{i}" for i in range(1, 42)]
actuator_vars = [f"xmv_{i}" for i in range(1, 12)]

# --------------------------------------
# 1. Información general de cada dataset
# --------------------------------------
for name, df in datasets.items():
    print(f"\n=== {name} ===")
    print("Tamaño:", df.shape)
    print("Número de simulaciones:", df['simulationRun'].nunique())
    print("Rango de muestras:", df['sample'].min(), "-", df['sample'].max())
    if 'faultNumber' in df.columns:
        print("Fallos presentes:", sorted(df['faultNumber'].unique()))
        if name.startswith("Faulty"):
            print("Conteo de fallos:")
            print(df['faultNumber'].value_counts().sort_index())

# --------------------------------------
# 2. Estadísticas descriptivas generales
# --------------------------------------
for name, df in datasets.items():
    print(f"\n=== Estadísticas generales: {name} ===")
    print(df[process_vars + actuator_vars].describe().T[['mean','std','min','25%','50%','75%','max']])

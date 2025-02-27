#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# Carga de los archivos CSV directamente desde el entorno de Jupyterhub
# file_path_region_0 = 'geo_data_0.csv'
# file_path_region_1 = 'geo_data_1.csv'
# file_path_region_2 = 'geo_data_2.csv'

file_path_region_0 = '/datasets/geo_data_0.csv'
file_path_region_1 = '/datasets/geo_data_1.csv'
file_path_region_2 = '/datasets/geo_data_2.csv'

# Carga de Los datos en DataFrames
data_region_0 = pd.read_csv(file_path_region_0)
data_region_1 = pd.read_csv(file_path_region_1)
data_region_2 = pd.read_csv(file_path_region_2)

# primeras filas de cada DataFrame para verificar
print("Región 0:")
print(data_region_0.head(), "\n")

print("Región 1:")
print(data_region_1.head(), "\n")

print("Región 2:")
print(data_region_2.head())


# In[3]:


# 1. Descripción de las características
print("Descripción estadística de los datos - Región 0:")
print(data_region_0.describe(), "\n")

print("Descripción estadística de los datos - Región 1:")
print(data_region_1.describe(), "\n")

print("Descripción estadística de los datos - Región 2:")
print(data_region_2.describe(), "\n")

# 2. Distribuciones de las características
def plot_distributions(data, region_name):
    plt.figure(figsize=(16, 10))
    plt.suptitle(f"Distribuciones de características - Región {region_name}", fontsize=16)

    plt.subplot(2, 2, 1)
    sns.histplot(data['f0'], kde=True, color='blue')
    plt.title('Distribución de f0')

    plt.subplot(2, 2, 2)
    sns.histplot(data['f1'], kde=True, color='green')
    plt.title('Distribución de f1')

    plt.subplot(2, 2, 3)
    sns.histplot(data['f2'], kde=True, color='red')
    plt.title('Distribución de f2')

    plt.subplot(2, 2, 4)
    sns.histplot(data['product'], kde=True, color='purple')
    plt.title('Distribución de reservas (product)')

    plt.tight_layout()
    plt.show()

# Graficar las distribuciones para cada región
plot_distributions(data_region_0, "0")
plot_distributions(data_region_1, "1")
plot_distributions(data_region_2, "2")

# 3. Matriz de correlación
def plot_correlation_matrix(data, region_name):
    plt.figure(figsize=(8, 6))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Matriz de correlación - Región {region_name}')
    plt.show()

# Matriz de correlación para cada región
plot_correlation_matrix(data_region_0, "0")
plot_correlation_matrix(data_region_1, "1")
plot_correlation_matrix(data_region_2, "2")

# 4. Boxplots para identificar outliers
def plot_boxplots(data, region_name):
    plt.figure(figsize=(16, 5))
    plt.suptitle(f"Boxplots de características - Región {region_name}", fontsize=16)

    plt.subplot(1, 4, 1)
    sns.boxplot(data=data, x='f0', color='blue')
    plt.title('Boxplot de f0')

    plt.subplot(1, 4, 2)
    sns.boxplot(data=data, x='f1', color='green')
    plt.title('Boxplot de f1')

    plt.subplot(1, 4, 3)
    sns.boxplot(data=data, x='f2', color='red')
    plt.title('Boxplot de f2')

    plt.subplot(1, 4, 4)
    sns.boxplot(data=data, x='product', color='purple')
    plt.title('Boxplot de reservas (product)')

    plt.tight_layout()
    plt.show()

# Graficar de los boxplots para cada región
plot_boxplots(data_region_0, "0")
plot_boxplots(data_region_1, "1")
plot_boxplots(data_region_2, "2")


# In[4]:


# Función para entrenar el modelo, predecir y calcular métricas
def train_and_evaluate(data, region_name):
    # Dividición de los datos en características (X) y objetivo (y)
    X = data[['f0', 'f1', 'f2']]
    y = data['product']
    
    # Dividición del conjunto de datos en conjunto de entrenamiento y validación (75%-25%)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Entrenamiento del modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones en el conjunto de validación
    predictions = model.predict(X_valid)
    
    # CalculO el RMSE
    rmse = mean_squared_error(y_valid, predictions, squared=False)
    
    # resultados
    print(f"Región {region_name}:")
    print(f"RMSE: {rmse}")
    print(f"Volumen medio de reservas predicho: {predictions.mean()} miles de barriles\n")
    
    # Retornar el modelo, las predicciones y el RMSE
    return model, predictions, y_valid, rmse

# Entrenamiento y evaluación para cada región
model_region_0, predictions_region_0, y_valid_region_0, rmse_region_0 = train_and_evaluate(data_region_0, "0")
model_region_1, predictions_region_1, y_valid_region_1, rmse_region_1 = train_and_evaluate(data_region_1, "1")
model_region_2, predictions_region_2, y_valid_region_2, rmse_region_2 = train_and_evaluate(data_region_2, "2")


# In[5]:


# Parámetros de negocio
pozos = 200  # Número de pozos a desarrollar
precio_por_barril = 4.5  # Precio de cada barril en USD
umbral_minimo = 111.1  # Umbral mínimo de barriles para evitar pérdidas

# Función para calcular la ganancia potencial de los 200 mejores pozos
def calcular_ganancias(predictions, region_name):
    # Seleccionar los 200 pozos con las predicciones más altas utilizando la función np.sort () para ordenar las predicciones de mayor a menor 
    mejores_pozos = np.sort(predictions)[-pozos:] # seleccion de los 200 pozos con valores mas altos.
    
    # Calculo del volumen total de reservas (en miles de barriles)
    volumen_total = mejores_pozos.sum()
    
    # Calculo de la ganancia total en dólares
    ganancia_total = volumen_total * precio_por_barril * 1000  # Convercción de miles de barriles a barriles
    
    # Resultados
    print(f"Región {region_name}:")
    print(f"Volumen total de reservas predicho para los 200 mejores pozos: {volumen_total} miles de barriles")
    print(f"Ganancia total estimada: ${ganancia_total:,.2f} USD\n")
    
    # Retornar la ganancia total
    return ganancia_total

# Calculo de la ganancias para las tres regiones
ganancia_region_0 = calcular_ganancias(predictions_region_0, "0")
ganancia_region_1 = calcular_ganancias(predictions_region_1, "1")
ganancia_region_2 = calcular_ganancias(predictions_region_2, "2")


# In[6]:


# Parámetros de negocio
pozos = 200
precio_por_barril = 4.5  # USD por barril
bootstrap_iterations = 1000  # Número de iteraciones para el bootstrapping
budget = 100_000_000  # USD

# Función para seleccionar los 200 mejores pozos y realizar el bootstrapping de ganancias
def bootstrapping_mejores_pozos(predictions, region_name):
    # Seleccionar los 200 mejores pozos (los que tienen las mayores predicciones de reservas)
    mejores_pozos = np.sort(predictions)[-pozos:]
    
    np.random.seed(42)  # Fijamos la semilla para reproducibilidad
    valores_muestras = []  # Lista para almacenar las ganancias de cada muestra
    
    for i in range(bootstrap_iterations):
        # Tomar una muestra aleatoria con reemplazo de los 200 mejores pozos seleccionados
        muestra = np.random.choice(mejores_pozos, size=pozos, replace=True)
        
        # Calcular el volumen total de la muestra
        volumen_muestra = muestra.sum()
        
        # Calcular las ganancias para la muestra
        ganancia_muestra = volumen_muestra * precio_por_barril * 1000  # Convertir a barriles y calcular ganancia
        valores_muestras.append(ganancia_muestra)
    
    # Convertir a numpy array para análisis estadístico
    valores_muestras = np.array(valores_muestras)
    
    # Calcular el beneficio promedio
    beneficio_promedio = valores_muestras.mean()
    
    # Calcular el intervalo de confianza del 95%
    intervalo_confianza = np.percentile(valores_muestras, [2.5, 97.5])
    
    # Calcular el riesgo de pérdidas (casos donde la ganancia es menor que el presupuesto)
    probabilidad_perdida = (valores_muestras < budget).mean() * 100  # Convertimos a porcentaje
    
    # Mostrar resultados
    print(f"Región {region_name}:")
    print(f"Beneficio promedio: ${beneficio_promedio:,.2f} USD")
    print(f"Intervalo de confianza del 95%: ${intervalo_confianza[0]:,.2f} - ${intervalo_confianza[1]:,.2f} USD")
    print(f"Riesgo de pérdidas: {probabilidad_perdida:.2f}%\n")
    
    # Retornar los valores clave
    return beneficio_promedio, intervalo_confianza, probabilidad_perdida

# Realizar el bootstrapping y calcular riesgos para cada región
beneficio_promedio_0, intervalo_0, perdida_0 = bootstrapping_mejores_pozos(predictions_region_0, "0")
beneficio_promedio_1, intervalo_1, perdida_1 = bootstrapping_mejores_pozos(predictions_region_1, "1")
beneficio_promedio_2, intervalo_2, perdida_2 = bootstrapping_mejores_pozos(predictions_region_2, "2")


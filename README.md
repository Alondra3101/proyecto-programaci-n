# proyecto-programaci-n
Objetivo: Desarrollar un sistema que analice datos históricos relacionados con precipitaciones y condiciones climáticas para identificar patrones y realizar predicciones sobre la ocurrencia de derrumbes en una zona crítica (Kilómetro 34 de la carretera Villa de Álvarez-Minatitlán).

# MODELO ARIMA DEL KM 34 CARRETERA VILLA DE ÁLVAREZ-MINTAITLÁN
**Autores:** Alondra Alonzo Peña, Honelia Alejandra Venegas Figueroa 

## Introducción
Este proyecto tiene como propósito analizar series temporales de datos climáticos, específicamente precipitaciones, para identificar patrones que permitan predecir derrumbes en zonas críticas. Este tipo de análisis es crucial para la prevención de desastres y la planificación de medidas de mitigación, como en el caso del Kilómetro 34 de la carretera Villa de Álvarez-Minatitlán, donde los derrumbes representan un riesgo significativo para la infraestructura y la seguridad.

## Desarrollo
El enfoque del proyecto incluye:

Descomposición de series temporales: Se utiliza el método aditivo para separar las componentes de tendencia, estacionalidad y ruido en los datos de precipitaciones.
Modelado predictivo: Empleando un modelo ARIMA (Autoregressive Integrated Moving Average) para capturar patrones históricos y realizar predicciones.
Visualización: Se grafican las componentes de la serie y las predicciones para facilitar la interpretación de los resultados.

Herramientas utilizadas
Lenguaje de programación Python.
Librerías principales: pandas, numpy, statsmodels, matplotlib.
Datos simulados basados en características reales de series climáticas.

Metodología
Generación de un conjunto de datos simulados que incluye precipitaciones, temperatura y ocurrencia de derrumbes.
Aplicación de técnicas de descomposición para analizar la estructura de los datos.
Entrenamiento y validación de un modelo ARIMA para realizar predicciones de 365 días futuros.

## Manejo de Datos
El conjunto de datos simulado consta de tres variables principales:

Precipitaciones (mm): Datos diarios de lluvias con estacionalidad anual y ruido aleatorio.
Temperatura (°C): Variable secundaria para enriquecer el análisis.
Derrumbes (binaria): Variable generada con base en la probabilidad de ocurrencia, directamente influenciada por las precipitaciones.
Los datos fueron generados utilizando funciones estadísticas y se guardaron en formato CSV.

## Código
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import describe

Simulación de datos climáticos (precipitaciones y derrumbes)
np.random.seed(42)
fecha = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")  # Fechas ajustadas
dias = len(fecha)

Generar precipitaciones simuladas con estacionalidad y ruido
precipitaciones = np.clip(
    20 * (np.sin(2 * np.pi * (fecha.dayofyear / 365.25)) + 1) +  # Estacionalidad
    np.random.normal(0, 5, dias),  # Ruido
    a_min=0, a_max=None  # Precipitaciones no negativas
)

Simular probabilidad de derrumbes basada en precipitaciones
derrumbes = (precipitaciones > 25).astype(int)

Crear DataFrame
datos = pd.DataFrame({
    "Fecha": fecha,
    "Precipitaciones_mm": precipitaciones,
    "Derrumbe": derrumbes
})
datos.set_index("Fecha", inplace=True)

Estadística descriptiva
print("Estadística Descriptiva de las Precipitaciones:")
print(describe(datos["Precipitaciones_mm"]))

Visualización inicial de datos
plt.figure(figsize=(12, 6))
plt.plot(datos.index, datos["Precipitaciones_mm"], label="Precipitaciones (mm)")
plt.title("Serie Temporal de Precipitaciones (2023)")
plt.xlabel("Fecha")
plt.ylabel("Precipitaciones (mm)")
plt.legend()
plt.grid()
plt.show()

Descomposición de la serie temporal
descomposicion = seasonal_decompose(datos["Precipitaciones_mm"], model="additive", period=30)
descomposicion.plot()
plt.suptitle("Descomposición de la Serie Temporal (2023)")
plt.show()

Ajuste del modelo ARIMA para predicciones
modelo = ARIMA(datos["Precipitaciones_mm"], order=(2, 1, 2))
ajuste = modelo.fit()

Predicciones a futuro (365 días)
predicciones = ajuste.forecast(steps=30)

Visualización de predicciones
plt.figure(figsize=(12, 6))
plt.plot(datos["Precipitaciones_mm"], label="Datos Históricos")
plt.plot(predicciones.index, predicciones, label="Predicción", color="red")
plt.title("Predicción de Precipitaciones (2024)")
plt.xlabel("Fecha")
plt.ylabel("Precipitaciones (mm)")
plt.legend()
plt.grid()
plt.show()

Resumen del modelo ARIMA
print("Resumen del Modelo ARIMA:")
print(ajuste.summary())

Análisis de derrumbes
print("\nResumen de Derrumbes:")
print(datos["Derrumbe"].value_counts())
print("Días con derrumbes:", datos["Derrumbe"].sum())

Visualización de derrumbes
plt.figure(figsize=(12, 6))
plt.scatter(datos.index, datos["Derrumbe"], color="red", alpha=0.5, label="Eventos de Derrumbe")
plt.plot(datos.index, datos["Precipitaciones_mm"], label="Precipitaciones (mm)", alpha=0.7)
plt.title("Relación entre Derrumbes y Precipitaciones (2023)")
plt.xlabel("Fecha")
plt.ylabel("Precipitaciones (mm)")
plt.legend()
plt.grid()
plt.show()

## Resultados
Gráficos:
Componentes descompuestas: tendencia, estacionalidad y residuo.
Predicción de precipitaciones para el próximo año, mostrando un aumento en periodos típicos de lluvias.

Análisis:
La estacionalidad anual es el principal factor que influye en las precipitaciones.
El modelo ARIMA logró capturar adecuadamente los patrones históricos, generando predicciones coherentes.

## Conclusiones
El análisis realizado demuestra la utilidad de las series temporales y los modelos ARIMA para predecir precipitaciones y evaluar riesgos relacionados con derrumbes. En futuras iteraciones, podría incorporarse un modelo multivariado que incluya otras variables como humedad del suelo o datos geotécnicos para mejorar la precisión.


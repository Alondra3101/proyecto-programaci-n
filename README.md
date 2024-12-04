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

#CODIGO MODIFICADO 
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np

# Coordenadas del kilómetro 34 (en grados decimales)
latitude = 19.390805  # 19°23'26.9"N
longitude = -103.95575  # 103°57'20.7"W

# Simulación de datos para el derrumbe
def simulate_terrain(lat, lon, buffer=0.01):
    """
    Simula un terreno antes y después de un derrumbe basado en un área rectangular.
    :param lat: Latitud del centro.
    :param lon: Longitud del centro.
    :param buffer: Tamaño del área a analizar (en grados).
    :return: Terreno original, terreno afectado y coordenadas.
    """
    # Crear una cuadrícula de puntos
    x = np.linspace(lon - buffer, lon + buffer, 100)
    y = np.linspace(lat - buffer, lat + buffer, 100)
    x, y = np.meshgrid(x, y)

    # Simular terreno antes del derrumbe (elevación en metros)
    z_original = 100 + 20 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

    # Simular terreno después del derrumbe
    z_afectado = z_original.copy()
    z_afectado[40:60, 40:60] -= 15  # Simular una caída de 15 metros en el centro

    return x, y, z_original, z_afectado

# Generar datos simulados
x, y, z_original, z_afectado = simulate_terrain(latitude, longitude)

# --- Visualización interactiva con Folium ---
def create_interactive_map(lat, lon):
    """
    Crea un mapa interactivo que muestra la ubicación del kilómetro 34 y su entorno.
    """
    mapa = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker(
        [lat, lon],
        popup="Kilómetro 34 - Carretera Villa de Álvarez - Minatitlán",
        tooltip="Derrumbe",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(mapa)
    return mapa

# --- Visualización del modelo 3D ---
def plot_terrain_3d(x, y, z_original, z_afectado):
    """
    Genera un gráfico 3D del terreno antes y después del derrumbe.
    """
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Terreno original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, z_original, cmap='terrain', edgecolor='none')
    ax1.set_title('Terreno Original')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax1.set_zlabel('Elevación (m)')

    # Subplot 2: Terreno afectado
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z_afectado, cmap='terrain', edgecolor='none')
    ax2.set_title('Terreno Afectado')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.set_zlabel('Elevación (m)')

    plt.tight_layout()
    plt.show()

# --- Cálculo del volumen del derrumbe ---
def calculate_volume(z_original, z_afectado, cell_size=10):
    """
    Calcula el volumen desplazado en el terreno.
    :param z_original: Elevación original.
    :param z_afectado: Elevación afectada.
    :param cell_size: Tamaño de la celda (en metros).
    :return: Volumen desplazado (en metros cúbicos).
    """
    diff = z_original - z_afectado
    volume = np.sum(diff[diff > 0]) * cell_size**2  # Solo sumar diferencias positivas
    return volume

# Crear mapa interactivo
mapa = create_interactive_map(latitude, longitude)
mapa.save("mapa_interactivo.html")  # Guarda el mapa en un archivo HTML
print("Mapa interactivo creado. Ábrelo con un navegador: 'mapa_interactivo.html'")

# Graficar el terreno en 3D
plot_terrain_3d(x, y, z_original, z_afectado)

# Calcular y mostrar el volumen desplazado
volumen = calculate_volume(z_original, z_afectado)
print(f"Volumen de tierra desplazada: {volumen:.2f} metros cúbicos")

## Resultados
Gráficos:
Componentes descompuestas: tendencia, estacionalidad y residuo.
Predicción de precipitaciones para el próximo año, mostrando un aumento en periodos típicos de lluvias.

Análisis:
La estacionalidad anual es el principal factor que influye en las precipitaciones.
El modelo ARIMA logró capturar adecuadamente los patrones históricos, generando predicciones coherentes.

## Conclusiones
El análisis realizado demuestra la utilidad de las series temporales y los modelos ARIMA para predecir precipitaciones y evaluar riesgos relacionados con derrumbes. En futuras iteraciones, podría incorporarse un modelo multivariado que incluya otras variables como humedad del suelo o datos geotécnicos para mejorar la precisión.


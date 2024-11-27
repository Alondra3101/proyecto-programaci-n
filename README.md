# proyecto-programaci-n
Objetivo: Desarrollar un sistema que analice datos históricos relacionados con precipitaciones y condiciones climáticas para identificar patrones y realizar predicciones sobre la ocurrencia de derrumbes en una zona crítica (Kilómetro 34 de la carretera Villa de Álvarez-Minatitlán).

# Título del Proyecto
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

## Resultados
Gráficos:
Componentes descompuestas: tendencia, estacionalidad y residuo.
Predicción de precipitaciones para el próximo año, mostrando un aumento en periodos típicos de lluvias.

Análisis:
La estacionalidad anual es el principal factor que influye en las precipitaciones.
El modelo ARIMA logró capturar adecuadamente los patrones históricos, generando predicciones coherentes.

## Conclusiones
El análisis realizado demuestra la utilidad de las series temporales y los modelos ARIMA para predecir precipitaciones y evaluar riesgos relacionados con derrumbes. En futuras iteraciones, podría incorporarse un modelo multivariado que incluya otras variables como humedad del suelo o datos geotécnicos para mejorar la precisión.


/// Entrenamiento de al menos un modelo de aprendizaje automático.
/// Presentación de resultados preliminares mediante métricas acordes al tipo de problema.

#heading(level:1)[Modelo entrenado y resultados preliminares]
#heading(level:2)[Modelos implementados]
Se entrenaron tres modelos de aprendizaje automático con los siguientes parámetros:

- *Regresión Lineal*: Modelo baseline con regularización estándar
- *Random Forest*: Ensamble de 100 árboles de decisión con paralelización
- *Gradient Boosting*: Boosting secuencial con 100 estimadores

#heading(level:2)[División de datos y estrategia de validación]
- *Entrenamiento*: 43,842 muestras (80%)
- *Prueba*: 10,961 muestras (20%)
- *Estrategia*: División aleatoria estratificada (seed=42)
- *Características*: 35 variables predictoras procesadas

#heading(level:2)[Métricas de evaluación]
Para evaluar el rendimiento de los modelos de regresión se utilizaron:

- *R² (Coeficiente de determinación)*: Proporción de varianza explicada
- *MSE (Error cuadrático medio)*: Penalización cuadrática de errores grandes
- *MAE (Error absoluto medio)*: Medida robusta de error promedio

#heading(level:2)[Resultados experimentales]
Los modelos mostraron el siguiente rendimiento diferenciado:

#figure(
  caption: [Métricas de rendimiento comparativo de los tres modelos],
)[
  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    // inset: 10pt,
    // align: (left, right, right, right, right, right, right),
    table.header([*Modelo*], [*R² Train*], [*R² Test*], [*MSE Train*], [*MSE Test*], [*MAE Train*], [*MAE Test*]),
    [Regresión Lineal] , [0.2460] , [0.2187], [640,539], [656,706], [548.18], [545.68],
    [Random Forest], [*0.9375*], [*0.5588*], [53.124], [370.880], [112.42], [298.30],
    [Gradient Boosting], [0.3379], [0.3229], [562.h21], [569.143], [504.24], [509.49],
  )
]

#heading(level:2)[Análisis de resultados]

#heading(level:3)[Modelo ganador: Random Forest]
*Random Forest* emergió como el modelo superior con:
- *R² = 0.5588*: Explica 55.88\% de la variabilidad del flujo de calor
- *MSE = 370,880 W²/m⁴*: Error cuadrático moderado
- *MAE = 298.30 W/m²*: Error absoluto promedio aceptable

#heading(level:3)[Detección de sobreajuste]
El análisis revela *sobreajuste moderado en Random Forest*:
- $R^2 "Train" (0.9375) gt.double R^2 "Test" (0.5588)$
- Diferencia de 0.38 puntos indica memorización de patrones de entrenamiento
- MSE aumenta significativamente de entrenamiento (53K) a prueba (371K)

#heading(level:3)[Rendimiento comparativo]
- *Regresión Lineal*: Rendimiento consistente pero limitado ($R^2 approx 0.22$)
- *Gradient Boosting*: Mejor generalización que Random Forest ($R^2 = 0.32$)
- *Random Forest*: Mayor capacidad predictiva pero con sobreajuste

Las visualizaciones implementadas permiten evaluar:
- *Predicciones vs. valores reales*: Correlación visual del modelo ganador
- *Análisis de residuos*: Detección de heterocedasticidad y patrones
- *Distribución de errores*: Verificación de normalidad en residuos
- *Importancia de características*: Ranking de variables meteorológicas influyentes

#heading(level:3)[Interpretación de resultados]
El *R² = 0.5588* indica que el modelo Random Forest captura efectivamente:
- Patrones de intercambio energético superficie-atmósfera
- Relaciones no-lineales entre variables meteorológicas
- Efectos de gradientes verticales de temperatura y viento
- Ciclos temporales diurnos y estacionales

El error absoluto medio de *298.30 W/m²* es razonable considerando:
- Rango de soil_heat: [-4,000, +4,132] W/m² (8,132 W/m² total)
- Error relativo: $tilde.op 7.3%$ del rango total
- Variabilidad natural alta en flujos de calor en ecosistemas montanos


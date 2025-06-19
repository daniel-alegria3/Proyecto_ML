#heading(level:1)[Conclusiones parciales y próximos pasos]

Durante esta primera etapa del proyecto se ha completado con éxito el análisis
exploratorio, preprocesamiento y entrenamiento de modelos para abordar el
problema de regresión orientado a la predicción del flujo de calor del suelo
(*`soil_heat`*) utilizando datos meteorológicos obtenidos de la Torre de
Gradiente del IGP en Junín, Perú.

El modelo con mejor desempeño lo obtuvo el modelo *Random Forest*, con un
$R^2$ en prueba de *0.5588*, una reducción significativa del
#emph([error cuadrático medio (MSE)]) en comparación con los otros modelos y un
#emph([MAE]) de 298.30, lo cual indica una mejor capacidad de ajuste sin
sobreajuste severo.

#heading(level:2, outlined: false, numbering: none)[Próximos pasos]
+ *Optimizar hiperparámetros* del modelo Random Forest con *GridSearchCV* o *RandomizedSearchCV*.
+ *Explorar modelos adicionales* como XGBoost, LightGBM o redes neuronales para comparar su rendimiento.
+ *Aplicar validación cruzada k-fold* para una evaluación más robusta del rendimiento.
+ *Analizar la importancia de características* del modelo para identificar qué variables tienen mayor influencia sobre el flujo de calor del suelo.
+ *Desarrollar visualizaciones interactivas* y/o un prototipo de sistema que permita visualizar predicciones en tiempo real con base en los datos atmosféricos.
+ Documentar todo el #emph([pipeline]) para facilitar la reproducibilidad del análisis.


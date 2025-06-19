/// Definición clara del tipo de problema: clasificación binaria, clasificación
/// multiclase o regresión.
///
/// Especificación del objetivo del modelo.

#heading(level:1)[Formulación del problema]

El problema se formula como una tarea de *regresión* donde el objetivo es
predecir el flujo de calor del suelo `soil_heat` utilizando las variables
meteorológicas medidas en diferentes niveles de la torre.

#heading(level:2)[Definición del objetivo]
El modelo busca establecer relaciones cuantitativas entre las condiciones
atmosféricas (temperatura, humedad, velocidad y dirección del viento) y el
intercambio de calor entre la superficie terrestre y la atmósfera. Esta
predicción es fundamental para:

- Estudios de balance energético en ecosistemas de alta montaña
- Análisis de procesos de intercambio superficie-atmósfera
- Estimación de flujos de calor sensible y latente
- Modelado microclimático en regiones andinas

La variable objetivo `soil_heat` representa el flujo de calor del suelo medido
a 8 cm de profundidad, expresado en W/m², donde valores positivos indican
transferencia de calor desde el suelo hacia la atmósfera y valores negativos
representan el flujo inverso.


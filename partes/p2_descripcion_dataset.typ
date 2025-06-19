/// Redactar la cantidad de filas y columnas. Describir los metadatos del dataset
/// y el dicionario de datos

#heading(level:1)[Descripción del dataset]

El dataset utilizado corresponde a mediciones de la estación Torre de Gradiente
del Laboratorio de Micro Física Atmosférica y Radiación (LAMAR) del Instituto
Geofísico del Perú (IGP), ubicada en Junín, Perú (-12.0399°S, -75.3207°W,
3316.78 m.s.n.m.). El dataset comprende registros meteorológicos desde mayo de
2018 hasta marzo de 2025.

El dataset contiene mediciones de una torre meteorológica de 30 metros de altura
con sensores distribuidos en diferentes niveles (2, 6, 12, 18, 24 y 29 metros)
que registran las principales variables atmosféricas con resolución temporal de
un minuto. Las mediciones incluyen temperatura y humedad relativa mediante
sondas HMP60 de Campbell Scientific, así como velocidad y dirección del viento
utilizando conjuntos Wind Sentry 03002.

#heading(level:2)[Estructura del dataset]
El dataset se compone de *25 columnas* principales que incluyen:

- *Variables temporales*: FECHA_CORTE, UBIGEO, year, month, day, hour
- *Temperatura del aire*: temp_n1 a temp_n6 (mediciones en 6 niveles, °C)
- *Velocidad del viento*: wind_n1 a wind_n6 (mediciones en 6 niveles, m/s)
- *Humedad relativa*: RH_n1 a RH_n6 (mediciones en 6 niveles, %)
- *Dirección del viento*: dir_wind_01 y dir_wind_02 (18m y 29m, grados)
- *Variable objetivo*: soil_heat (flujo de calor del suelo a 8 cm, W/m²)

Todas las variables numéricas se almacenan con precisión de cinco decimales, con
valores enteros positivos para viento y humedad, y posibles valores negativos
para temperatura y flujo de calor del suelo.


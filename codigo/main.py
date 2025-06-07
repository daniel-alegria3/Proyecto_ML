# Análisis de Machine Learning - Torre de Gradiente
# Predicción de flujo de calor del suelo usando variables meteorológicas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# 1. CARGA Y ANÁLISIS EXPLORATORIO DE DATOS
# ============================================================================

# Cargar datos - cambiar la ruta según tu archivo
CSV_PATH = "./IGP_EstacionTorreGradiente_2018-2025_Dataset_0.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")
except Exception as e:
    print(f"Error al cargar datos: {e}")

# Información básica del dataset
print(f"\nPeríodo de datos: {df['year'].min()}-{df['year'].max()}")
print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Verificar columnas disponibles
print(f"\nColumnas disponibles ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
display(df[numeric_cols].describe())

# Valores faltantes
print("\nValores faltantes:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Columna': missing_data.index,
    'Valores Faltantes': missing_data.values,
    'Porcentaje': missing_percent.values
})
missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Porcentaje', ascending=False)

if not missing_df.empty:
    display(missing_df)
else:
    print("No hay valores faltantes")

# ============================================================================
# 2. VISUALIZACIONES EXPLORATORIAS
# ============================================================================

fig = plt.figure(figsize=(20, 15))

# Distribución de la variable objetivo (soil_heat)
if 'soil_heat' in df.columns:
    plt.subplot(3, 3, 1)
    plt.hist(df['soil_heat'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución del Flujo de Calor del Suelo')
    plt.xlabel('Flujo de Calor (W/m²)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)

# Series temporales de temperatura
temp_cols = [col for col in df.columns if 'temp_n' in col]
if temp_cols:
    plt.subplot(3, 3, 2)
    for col in temp_cols[:3]:
        if col in df.columns:
            plt.plot(df.index[:1000], df[col][:1000], alpha=0.7, label=col)
    plt.title('Series Temporales de Temperatura (Primeros 1000 puntos)')
    plt.xlabel('Índice Temporal')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Boxplot de velocidades de viento
wind_cols = [col for col in df.columns if 'wind_n' in col]
if wind_cols:
    plt.subplot(3, 3, 3)
    wind_data = [df[col].dropna().values for col in wind_cols if col in df.columns]
    plt.boxplot(wind_data, labels=wind_cols)
    plt.title('Distribución de Velocidades de Viento por Nivel')
    plt.xlabel('Nivel de Medición')
    plt.ylabel('Velocidad del Viento (m/s)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# Humedad relativa por nivel
rh_cols = [col for col in df.columns if 'RH_n' in col]
if rh_cols:
    plt.subplot(3, 3, 4)
    rh_data = [df[col].dropna().values for col in rh_cols if col in df.columns]
    plt.boxplot(rh_data, labels=rh_cols)
    plt.title('Distribución de Humedad Relativa por Nivel')
    plt.xlabel('Nivel de Medición')
    plt.ylabel('Humedad Relativa (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# Correlación entre variables
plt.subplot(3, 3, 5)
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
           square=True, fmt='.2f', cbar_kws={'shrink': .8})
plt.title('Matriz de Correlación')

# Variación diurna promedio
if 'hour' in df.columns and 'soil_heat' in df.columns:
    plt.subplot(3, 3, 6)
    hourly_avg = df.groupby('hour')['soil_heat'].mean()
    plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
    plt.title('Variación Diurna Promedio del Flujo de Calor')
    plt.xlabel('Hora del Día')
    plt.ylabel('Flujo de Calor Promedio (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))

# Scatter plot: Temperatura vs Flujo de Calor
if 'temp_n1' in df.columns and 'soil_heat' in df.columns:
    plt.subplot(3, 3, 7)
    plt.scatter(df['temp_n1'], df['soil_heat'], alpha=0.5, s=1)
    plt.title('Temperatura vs Flujo de Calor del Suelo')
    plt.xlabel('Temperatura Nivel 1 (°C)')
    plt.ylabel('Flujo de Calor (W/m²)')
    plt.grid(True, alpha=0.3)

# Variación estacional
if 'month' in df.columns and 'soil_heat' in df.columns:
    plt.subplot(3, 3, 8)
    monthly_avg = df.groupby('month')['soil_heat'].mean()
    plt.bar(monthly_avg.index, monthly_avg.values, color='lightcoral', alpha=0.7)
    plt.title('Variación Estacional del Flujo de Calor')
    plt.xlabel('Mes')
    plt.ylabel('Flujo de Calor Promedio (W/m²)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))

# Distribución de dirección del viento
if 'dir_wind_01' in df.columns:
    plt.subplot(3, 3, 9)
    wind_dir = df['dir_wind_01'].dropna()
    plt.hist(wind_dir, bins=36, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribución de Dirección del Viento')
    plt.xlabel('Dirección del Viento (grados)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================

# Crear una copia para trabajar
df_processed = df.copy()

# Construir un índice temporal (hora promedio del registro)
df_processed['timestamp'] = pd.to_datetime(dict(year=df_processed['year'],
                                           month=df_processed['month'],
                                           day=df_processed['day'],
                                           hour=df_processed['hour']))
# Ordenar por el timestamp
df_processed = df_processed.sort_values('timestamp').reset_index(drop=True)
df_processed.set_index('timestamp', inplace=True)

# Aplicar interpolación por tiempo
df_processed = df_processed.interpolate(method='time')

# Verificación: revisar valores faltantes después de la interpolación
missing_after = df_processed.isnull().sum()
print("Valores faltantes después de la interpolación:\n", missing_after)

# Crear características temporales
if 'hour' in df_processed.columns:
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)

if 'month' in df_processed.columns:
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)

# Crear gradientes de temperatura
temp_cols_available = [col for col in temp_cols if col in df_processed.columns]
if len(temp_cols_available) >= 2:
    for i in range(len(temp_cols_available)-1):
        gradient_name = f'temp_gradient_{i+1}_{i+2}'
        df_processed[gradient_name] = df_processed[temp_cols_available[i+1]] - df_processed[temp_cols_available[i]]

# Crear promedios y desviaciones estándar por grupo de variables
if temp_cols_available:
    df_processed['temp_mean'] = df_processed[temp_cols_available].mean(axis=1)
    df_processed['temp_std'] = df_processed[temp_cols_available].std(axis=1)

wind_cols_available = [col for col in wind_cols if col in df_processed.columns]
if wind_cols_available:
    df_processed['wind_mean'] = df_processed[wind_cols_available].mean(axis=1)
    df_processed['wind_std'] = df_processed[wind_cols_available].std(axis=1)

rh_cols_available = [col for col in rh_cols if col in df_processed.columns]
if rh_cols_available:
    df_processed['rh_mean'] = df_processed[rh_cols_available].mean(axis=1)
    df_processed['rh_std'] = df_processed[rh_cols_available].std(axis=1)

# Definir variable objetivo y características
target = 'soil_heat'
if target not in df_processed.columns:
    # Si no existe soil_heat, usar temp_n1 como ejemplo
    target = 'temp_n1'
    print(f"Advertencia: 'soil_heat' no encontrada, usando '{target}' como variable objetivo")

# Seleccionar características predictoras
feature_cols = []

# Variables temporales
time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
feature_cols.extend([col for col in time_features if col in df_processed.columns])

# Variables meteorológicas originales
met_features = temp_cols_available + wind_cols_available + rh_cols_available
feature_cols.extend(met_features)

# Variables de dirección del viento
wind_dir_cols = [col for col in df_processed.columns if 'dir_wind' in col]
feature_cols.extend(wind_dir_cols[:2])

# Variables derivadas
derived_features = [col for col in df_processed.columns if
                   ('gradient' in col or 'mean' in col or 'std' in col)]
feature_cols.extend(derived_features)

# Eliminar duplicados y filtrar columnas existentes
feature_cols = list(set([col for col in feature_cols if col in df_processed.columns]))

print(f"Características seleccionadas: {len(feature_cols)}")
print(f"Variable objetivo: {target}")

# Preparar matrices X e y
X = df_processed[feature_cols].copy()
y = df_processed[target].copy()

# Eliminar filas con valores faltantes en la variable objetivo
valid_idx = ~y.isnull()
X = X[valid_idx]
y = y[valid_idx]

print(f"Datos después de limpieza: {X.shape[0]} muestras, {X.shape[1]} características")

# Imputar valores faltantes en características
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir de vuelta a DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print(f"División de datos:")
print(f"- Entrenamiento: {X_train.shape[0]} muestras")
print(f"- Prueba: {X_test.shape[0]} muestras")

# ============================================================================
# 4. ENTRENAMIENTO DE MODELOS
# ============================================================================

print("\nEntrenando modelos de machine learning...")

# Definir modelos
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Almacenar resultados
results = {}
trained_models = {}

for name, model in models.items():
    print(f"Entrenando {name}...")

    # Entrenar modelo
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    # Calcular métricas
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    # Guardar resultados
    trained_models[name] = model
    results[name] = {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_pred': train_pred, 'test_pred': test_pred
    }

    print(f"  R²: {test_r2:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

print("Entrenamiento completado")

# ============================================================================
# 5. EVALUACIÓN Y RESULTADOS
# ============================================================================

# Crear tabla de resultados
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'R² Train': [results[model]['train_r2'] for model in results.keys()],
    'R² Test': [results[model]['test_r2'] for model in results.keys()],
    'MSE Train': [results[model]['train_mse'] for model in results.keys()],
    'MSE Test': [results[model]['test_mse'] for model in results.keys()],
    'MAE Train': [results[model]['train_mae'] for model in results.keys()],
    'MAE Test': [results[model]['test_mae'] for model in results.keys()]
})

print("\nMétricas de rendimiento:")
display(results_df.round(4))

# Identificar mejor modelo
best_model_name = results_df.loc[results_df['R² Test'].idxmax(), 'Modelo']
best_r2 = results_df.loc[results_df['R² Test'].idxmax(), 'R² Test']

print(f"\nMejor modelo: {best_model_name}")
print(f"R² en prueba: {best_r2:.4f}")
print(f"MSE en prueba: {results[best_model_name]['test_mse']:.4f}")
print(f"MAE en prueba: {results[best_model_name]['test_mae']:.4f}")

# ============================================================================
# 6. VISUALIZACIONES DE RESULTADOS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Comparación de R²
models_list = list(results.keys())
train_r2_list = [results[model]['train_r2'] for model in models_list]
test_r2_list = [results[model]['test_r2'] for model in models_list]

x = np.arange(len(models_list))
width = 0.35

axes[0, 0].bar(x - width/2, train_r2_list, width, label='Entrenamiento', alpha=0.8)
axes[0, 0].bar(x + width/2, test_r2_list, width, label='Prueba', alpha=0.8)
axes[0, 0].set_xlabel('Modelos')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('Comparación de R² por Modelo')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models_list, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Predicciones vs Valores Reales (mejor modelo)
test_pred_best = results[best_model_name]['test_pred']

axes[0, 1].scatter(y_test, test_pred_best, alpha=0.6, s=20)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Valores Reales')
axes[0, 1].set_ylabel('Predicciones')
axes[0, 1].set_title(f'Predicciones vs Reales - {best_model_name}')
axes[0, 1].grid(True, alpha=0.3)

# Residuos
residuals = y_test - test_pred_best
axes[1, 0].scatter(test_pred_best, residuals, alpha=0.6, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicciones')
axes[1, 0].set_ylabel('Residuos')
axes[1, 0].set_title(f'Análisis de Residuos - {best_model_name}')
axes[1, 0].grid(True, alpha=0.3)

# Distribución de errores
axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Residuos')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de Residuos')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Importancia de características (si disponible)
best_model = trained_models[best_model_name]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X_train.columns

    # Ordenar por importancia
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f'Importancia de Características - {best_model_name}')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]])
    plt.xticks(range(min(20, len(importances))),
              [feature_names[i] for i in indices[:20]],
              rotation=45, ha='right')
    plt.xlabel('Características')
    plt.ylabel('Importancia')
    plt.tight_layout()
    plt.show()

print("\nAnálisis completado")


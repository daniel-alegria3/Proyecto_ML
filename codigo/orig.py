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

class TorreGradienteML:
    def __init__(self, csv_path):
        """
        Inicializa la clase para análisis de ML de datos de Torre de Gradiente

        Definición del problema:
        - Tipo: REGRESIÓN
        - Objetivo: Predecir el flujo de calor del suelo (soil_heat) basado en variables meteorológicas
        - Variables predictoras: temperatura, humedad, viento en diferentes niveles
        """
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_and_explore_data(self):
        """Carga y realiza análisis exploratorio de datos"""
        print("=" * 60)
        print("ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 60)

        # Cargar datos
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Datos cargados exitosamente: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return

        # Información básica del dataset
        print(f"\n📊 INFORMACIÓN GENERAL:")
        print(f"   • Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
        print(f"   • Período: {self.df['year'].min()}-{self.df['year'].max()}")
        print(f"   • Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Verificar columnas esperadas según metadatos
        expected_cols = ['FECHA_CORTE', 'UBIGEO', 'year', 'month', 'day',
                        'hour', 'temp_n1', 'temp_n2', 'temp_n3', 'temp_n4', 'temp_n5',
                        'temp_n6', 'wind_n1', 'wind_n2', 'wind_n3', 'wind_n4', 'wind_n5',
                        'wind_n6', 'RH_n1', 'RH_n2', 'RH_n3', 'RH_n4', 'RH_n5', 'RH_n6',
                        'dir_wind_01', 'dir_wind_02', 'soil_heat']

        print(f"\n📋 COLUMNAS DISPONIBLES:")
        available_cols = list(self.df.columns)
        for i, col in enumerate(available_cols):
            print(f"   {i+1:2d}. {col}")

        # Estadísticas descriptivas
        print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())

        # Valores faltantes
        print(f"\n🔍 VALORES FALTANTES:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Porcentaje': missing_percent.values
        })
        missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
        if not missing_df.empty:
            print(missing_df.to_string(index=False))
        else:
            print("   ✓ No hay valores faltantes")

        return self.df

    def create_visualizations(self):
        """Crea visualizaciones para análisis exploratorio"""
        if self.df is None:
            print("❌ Primero debe cargar los datos")
            return

        print("\n" + "=" * 60)
        print("VISUALIZACIONES DE ANÁLISIS EXPLORATORIO")
        print("=" * 60)

        # Configurar subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Distribución de la variable objetivo (soil_heat)
        if 'soil_heat' in self.df.columns:
            plt.subplot(3, 3, 1)
            plt.hist(self.df['soil_heat'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribución del Flujo de Calor del Suelo', fontsize=12, fontweight='bold')
            plt.xlabel('Flujo de Calor (W/m²)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)

        # 2. Series temporales de temperatura
        temp_cols = [col for col in self.df.columns if 'temp_n' in col]
        if temp_cols:
            plt.subplot(3, 3, 2)
            for col in temp_cols[:3]:  # Solo primeros 3 niveles
                if col in self.df.columns:
                    plt.plot(self.df.index[:1000], self.df[col][:1000], alpha=0.7, label=col)
            plt.title('Series Temporales de Temperatura (Primeros 1000 puntos)', fontsize=12, fontweight='bold')
            plt.xlabel('Índice Temporal')
            plt.ylabel('Temperatura (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 3. Boxplot de velocidades de viento
        wind_cols = [col for col in self.df.columns if 'wind_n' in col]
        if wind_cols:
            plt.subplot(3, 3, 3)
            wind_data = [self.df[col].dropna().values for col in wind_cols if col in self.df.columns]
            plt.boxplot(wind_data, labels=wind_cols)
            plt.title('Distribución de Velocidades de Viento por Nivel', fontsize=12, fontweight='bold')
            plt.xlabel('Nivel de Medición')
            plt.ylabel('Velocidad del Viento (m/s)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        # 4. Humedad relativa por nivel
        rh_cols = [col for col in self.df.columns if 'RH_n' in col]
        if rh_cols:
            plt.subplot(3, 3, 4)
            rh_data = [self.df[col].dropna().values for col in rh_cols if col in self.df.columns]
            plt.boxplot(rh_data, labels=rh_cols)
            plt.title('Distribución de Humedad Relativa por Nivel', fontsize=12, fontweight='bold')
            plt.xlabel('Nivel de Medición')
            plt.ylabel('Humedad Relativa (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        # 5. Correlación entre variables
        plt.subplot(3, 3, 5)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': .8})
        plt.title('Matriz de Correlación', fontsize=12, fontweight='bold')

        # 6. Variación diurna promedio
        if 'hour' in self.df.columns and 'soil_heat' in self.df.columns:
            plt.subplot(3, 3, 6)
            hourly_avg = self.df.groupby('hour')['soil_heat'].mean()
            plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
            plt.title('Variación Diurna Promedio del Flujo de Calor', fontsize=12, fontweight='bold')
            plt.xlabel('Hora del Día')
            plt.ylabel('Flujo de Calor Promedio (W/m²)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))

        # 7. Scatter plot: Temperatura vs Flujo de Calor
        if 'temp_n1' in self.df.columns and 'soil_heat' in self.df.columns:
            plt.subplot(3, 3, 7)
            plt.scatter(self.df['temp_n1'], self.df['soil_heat'], alpha=0.5, s=1)
            plt.title('Temperatura vs Flujo de Calor del Suelo', fontsize=12, fontweight='bold')
            plt.xlabel('Temperatura Nivel 1 (°C)')
            plt.ylabel('Flujo de Calor (W/m²)')
            plt.grid(True, alpha=0.3)

        # 8. Variación estacional
        if 'month' in self.df.columns and 'soil_heat' in self.df.columns:
            plt.subplot(3, 3, 8)
            monthly_avg = self.df.groupby('month')['soil_heat'].mean()
            plt.bar(monthly_avg.index, monthly_avg.values, color='lightcoral', alpha=0.7)
            plt.title('Variación Estacional del Flujo de Calor', fontsize=12, fontweight='bold')
            plt.xlabel('Mes')
            plt.ylabel('Flujo de Calor Promedio (W/m²)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, 13))

        # 9. Distribución de dirección del viento
        if 'dir_wind_01' in self.df.columns:
            plt.subplot(3, 3, 9)
            wind_dir = self.df['dir_wind_01'].dropna()
            plt.hist(wind_dir, bins=36, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Distribución de Dirección del Viento', fontsize=12, fontweight='bold')
            plt.xlabel('Dirección del Viento (grados)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("✓ Visualizaciones generadas exitosamente")
    def preprocess_data(self):
        """Limpieza y preparación del dataset"""
        if self.df is None:
            print("❌ Primero debe cargar los datos")
            return

        # Construir un índice temporal (hora promedio del registro)
        self.df['timestamp'] = pd.to_datetime(dict(year=self.df['year'],
                                            month=self.df['month'],
                                            day=self.df['day'],
                                            hour=self.df['hour']))

        # Ordenar por el timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        self.df.set_index('timestamp', inplace=True)

        # Aplicar interpolación por tiempo
        self.df = self.df.interpolate(method='time')

        # Verificación: revisar valores faltantes después de la interpolación
        missing_after = self.df.isnull().sum()
        print("Valores faltantes después de la interpolación:\n", missing_after)
        print("\n" + "=" * 60)

        print("METODOLOGÍA DE PREPROCESAMIENTO")
        print("=" * 60)

        # 1. Crear características temporales
        print("🔧 Creando características temporales...")
        if 'hour' in self.df.columns:
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        # 2. Crear gradientes de temperatura
        print("🔧 Calculando gradientes de temperatura...")
        temp_cols = [col for col in self.df.columns if 'temp_n' in col and col in self.df.columns]
        if len(temp_cols) >= 2:
            for i in range(len(temp_cols)-1):
                gradient_name = f'temp_gradient_{i+1}_{i+2}'
                self.df[gradient_name] = self.df[temp_cols[i+1]] - self.df[temp_cols[i]]

        # 3. Crear promedios de variables por nivel
        print("🔧 Calculando promedios de variables...")
        # Promedio de temperaturas
        temp_cols_available = [col for col in temp_cols if col in self.df.columns]
        if temp_cols_available:
            self.df['temp_mean'] = self.df[temp_cols_available].mean(axis=1)
            self.df['temp_std'] = self.df[temp_cols_available].std(axis=1)

        # Promedio de vientos
        wind_cols = [col for col in self.df.columns if 'wind_n' in col and col in self.df.columns]
        if wind_cols:
            self.df['wind_mean'] = self.df[wind_cols].mean(axis=1)
            self.df['wind_std'] = self.df[wind_cols].std(axis=1)

        # Promedio de humedad
        rh_cols = [col for col in self.df.columns if 'RH_n' in col and col in self.df.columns]
        if rh_cols:
            self.df['rh_mean'] = self.df[rh_cols].mean(axis=1)
            self.df['rh_std'] = self.df[rh_cols].std(axis=1)

        # 4. Selección de características para el modelo
        print("🔧 Seleccionando características para el modelo...")

        # Definir variable objetivo
        target = 'soil_heat'
        if target not in self.df.columns:
            # Si no existe soil_heat, usar temp_n1 como ejemplo
            target = 'temp_n1'
            print(f"⚠️  'soil_heat' no encontrada, usando '{target}' como variable objetivo")

        # Seleccionar características predictoras
        feature_cols = []

        # Variables temporales
        time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        feature_cols.extend([col for col in time_features if col in self.df.columns])

        # Variables meteorológicas originales
        met_features = temp_cols + wind_cols + rh_cols
        feature_cols.extend([col for col in met_features if col in self.df.columns])

        # Variables de dirección del viento
        wind_dir_cols = [col for col in self.df.columns if 'dir_wind' in col]
        feature_cols.extend(wind_dir_cols[:2])  # Solo primeras 2

        # Variables derivadas
        derived_features = [col for col in self.df.columns if
                          ('gradient' in col or 'mean' in col or 'std' in col)]
        feature_cols.extend(derived_features)

        # Eliminar duplicados y filtrar columnas existentes
        feature_cols = list(set([col for col in feature_cols if col in self.df.columns]))

        print(f"📋 Características seleccionadas ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {col}")

        # 5. Preparar datos para ML
        print(f"\n🎯 Variable objetivo: {target}")

        # Crear matrices X e y
        X = self.df[feature_cols].copy()
        y = self.df[target].copy()

        # Eliminar filas con valores faltantes en y
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"📊 Datos después de limpieza: {X.shape[0]} muestras, {X.shape[1]} características")

        # Imputar valores faltantes en X
        print("🔧 Imputando valores faltantes...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)

        # División train/test
        print("🔧 Dividiendo datos en entrenamiento y prueba (80/20)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Normalización
        print("🔧 Normalizando características...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Convertir de vuelta a DataFrame
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=feature_cols, index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=feature_cols, index=self.X_test.index)

        print(f"✓ Preprocesamiento completado:")
        print(f"   • Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"   • Prueba: {self.X_test.shape[0]} muestras")
        print(f"   • Características: {self.X_train.shape[1]}")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_models(self):
        """Entrenamiento de modelos de aprendizaje automático"""
        if self.X_train is None:
            print("❌ Primero debe preprocesar los datos")
            return

        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DE MODELOS")
        print("=" * 60)

        # Definir modelos
        models_config = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        print(f"🤖 Entrenando {len(models_config)} modelos...")

        for name, model in models_config.items():
            print(f"\n🔄 Entrenando {name}...")

            # Entrenar modelo
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                train_pred = model.predict(self.X_train_scaled)
                test_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)

            # Calcular métricas
            train_mse = mean_squared_error(self.y_train, train_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)

            # Guardar modelo y resultados
            self.models[name] = model
            self.results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_pred': train_pred,
                'test_pred': test_pred
            }

            print(f"   ✓ R²: {test_r2:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

        print(f"\n✅ Entrenamiento completado para todos los modelos")

    def show_results(self):
        """Presentación de resultados mediante métricas"""
        if not self.results:
            print("❌ Primero debe entrenar los modelos")
            return

        print("\n" + "=" * 60)
        print("RESULTADOS PRELIMINARES")
        print("=" * 60)

        # Tabla de resultados
        results_df = pd.DataFrame({
            'Modelo': list(self.results.keys()),
            'R² Train': [self.results[model]['train_r2'] for model in self.results.keys()],
            'R² Test': [self.results[model]['test_r2'] for model in self.results.keys()],
            'MSE Train': [self.results[model]['train_mse'] for model in self.results.keys()],
            'MSE Test': [self.results[model]['test_mse'] for model in self.results.keys()],
            'MAE Train': [self.results[model]['train_mae'] for model in self.results.keys()],
            'MAE Test': [self.results[model]['test_mae'] for model in self.results.keys()]
        })

        print("📊 MÉTRICAS DE RENDIMIENTO:")
        print(results_df.round(4).to_string(index=False))

        # Identificar mejor modelo
        best_model = results_df.loc[results_df['R² Test'].idxmax(), 'Modelo']
        best_r2 = results_df.loc[results_df['R² Test'].idxmax(), 'R² Test']

        print(f"\n🏆 MEJOR MODELO: {best_model}")
        print(f"   • R² en prueba: {best_r2:.4f}")
        print(f"   • MSE en prueba: {self.results[best_model]['test_mse']:.4f}")
        print(f"   • MAE en prueba: {self.results[best_model]['test_mae']:.4f}")

        # Visualizaciones de resultados
        self.plot_results()

        # Importancia de características (si disponible)
        if best_model in ['Random Forest', 'Gradient Boosting']:
            self.plot_feature_importance(best_model)

        return results_df

    def plot_results(self):
        """Crear visualizaciones de resultados"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Comparación de R²
        models = list(self.results.keys())
        train_r2 = [self.results[model]['train_r2'] for model in models]
        test_r2 = [self.results[model]['test_r2'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        axes[0, 0].bar(x - width/2, train_r2, width, label='Entrenamiento', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='Prueba', alpha=0.8)
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Comparación de R² por Modelo')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Predicciones vs Valores Reales (mejor modelo)
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        test_pred = self.results[best_model]['test_pred']

        axes[0, 1].scatter(self.y_test, test_pred, alpha=0.6, s=20)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Valores Reales')
        axes[0, 1].set_ylabel('Predicciones')
        axes[0, 1].set_title(f'Predicciones vs Reales - {best_model}')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuos
        residuals = self.y_test - test_pred
        axes[1, 0].scatter(test_pred, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicciones')
        axes[1, 0].set_ylabel('Residuos')
        axes[1, 0].set_title(f'Análisis de Residuos - {best_model}')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Distribución de errores
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuos')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Residuos')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model_name):
        """Graficar importancia de características"""
        if model_name not in self.models:
            return

        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns

            # Ordenar por importancia
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title(f'Importancia de Características - {model_name}')
            plt.bar(range(min(20, len(importances))),
                   importances[indices[:20]])
            plt.xticks(range(min(20, len(importances))),
                      [feature_names[i] for i in indices[:20]],
                      rotation=45, ha='right')
            plt.xlabel('Características')
            plt.ylabel('Importancia')
            plt.tight_layout()
            plt.show()

    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE MACHINE LEARNING")
        print("🎯 PROBLEMA: Regresión para predicción de flujo de calor del suelo")
        print("📍 UBICACIÓN: Torre de Gradiente LAMAR, Junín, Perú")

        # Ejecutar todos los pasos
        self.load_and_explore_data()
        self.create_visualizations()
        self.preprocess_data()
        self.train_models()
        results = self.show_results()

        print("\n" + "=" * 60)
        print("RESUMEN EJECUTIVO")
        print("=" * 60)
        print("✅ Análisis exploratorio completado con visualizaciones")
        print("✅ Preprocesamiento realizado con ingeniería de características")
        print("✅ Múltiples modelos entrenados y evaluados")
        print("✅ Resultados preliminares presentados con métricas apropiadas")

        return results

# INSTRUCCIONES DE USO:
# ===================
# 1. Guarda este código en un archivo .py
# 2. Asegúrate de tener tu archivo CSV de datos
# 3. Ejecuta el siguiente código:

if __name__ == "__main__":
    # Cambia la ruta por la ubicación de tu archivo CSV
    CSV_PATH = "IGP.csv"  # ← CAMBIAR ESTA RUTA

    # Crear instancia y ejecutar análisis
    analyzer = TorreGradienteML(CSV_PATH)
    results = analyzer.run_complete_analysis()

    print(f"\n🎉 Análisis completado exitosamente!")
    print(f"📊 Revisa las visualizaciones y métricas generadas")


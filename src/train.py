import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Загрузка данных
df = pd.read_csv("fish_participant.csv")

# Разделяем признаки и целевую переменную
# Создаем производный признак (объем) для эксперимента E4
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
X = df.drop("Weight", axis=1)
y = df["Weight"]

# 2. Разбиение на Train и Test (фиксируем random_state, чтобы выборки не менялись)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Выделяем числовые и категориальные признаки
numeric_features = ["Length1", "Length2", "Length3", "Height", "Width", "Vproxy"]
categorical_features = ["Species"]

# 3. Создаем пайплайны для числовых и категориальных признаков
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 4. Объединяем их в ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. Функция для обучения и логирования в MLflow
def run_experiment(model_name, regressor, use_log_target=False):
    # Если используем логарифмирование целевой переменной (для борьбы с асимметрией)
    if use_log_target:
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1
        )
    
    # Собираем полный пайплайн
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    # Запускаем эксперимент в MLflow
    with mlflow.start_run(run_name=model_name):
        # Обучение
        pipeline.fit(X_train, y_train)
        
        # Предсказание
        y_pred = pipeline.predict(X_test)
        
        # Расчет метрик
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Логирование параметров и метрик
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("use_log_target", use_log_target)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        # Сохранение модели
        mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])

        print(f"[{model_name}] MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        return pipeline

# 6. Определяем модели для экспериментов (E0 - E3)
models = {
    "E0_Dummy_Mean": (DummyRegressor(strategy="mean"), False),
    "E1_Linear_Log": (LinearRegression(), True),
    "E2_Ridge_Log": (Ridge(alpha=1.0), True),
    "E3_RandomForest": (RandomForestRegressor(n_estimators=100, random_state=42), False),
    "E4_RandomForest_Volume": (RandomForestRegressor(n_estimators=100, random_state=42), False)
}

# 7. Запускаем все эксперименты
if __name__ == "__main__":
    for name, (model, use_log) in models.items():
        run_experiment(model_name=name, regressor=model, use_log_target=use_log)
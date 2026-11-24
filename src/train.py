import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
import joblib
import os
import subprocess
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Загрузка данных и создание признака Vproxy (объем)
df = pd.read_csv("fish_participant.csv")
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]

X = df.drop("Weight", axis=1)
y = df["Weight"]

# 2. Разбиение на Train и Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Базовые признаки (для E0, E1, E3)
base_numeric = ["Length1", "Length2", "Length3", "Height", "Width"]
# Признаки с объемом (для E2 и E4)
extended_numeric = base_numeric + ["Vproxy"]
categorical_features = ["Species"]

# 3. Загружаем конфигурацию из YAML
with open("configs/params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Создаем папку для сохранения моделей, если ее нет
os.makedirs("models", exist_ok=True)

# 4. Функция для создания пайплайна
def create_pipeline(numeric_features, regressor, use_log_target=False):
    if use_log_target:
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1
        )
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

# 5. Функция для обучения и логирования
def run_experiment(model_name, regressor, numeric_features, param_grid=None, use_log_target=False):
    pipeline = create_pipeline(numeric_features, regressor, use_log_target)
    
    # Если передан param_grid, используем GridSearchCV (Кросс-валидация)
    if param_grid:
        # Динамически выбираем префикс в зависимости от обертки
        prefix = "regressor__regressor__" if use_log_target else "regressor__"
        grid_params = {f"{prefix}{k}": v for k, v in param_grid.items()}
        model_to_fit = GridSearchCV(pipeline, grid_params, cv=3, scoring='neg_mean_squared_error')
    else:
        model_to_fit = pipeline

    with mlflow.start_run(run_name=model_name):
        # Получаем хэш коммита и версию Python для тегов MLflow
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception:
            commit_hash = "unknown"
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.set_tag("python_version", sys.version.split()[0])

        model_to_fit.fit(X_train, y_train)
        
        # Если это GridSearchCV, достаем лучшую модель
        best_model = model_to_fit.best_estimator_ if param_grid else model_to_fit
        
        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("use_log_target", use_log_target)
        mlflow.log_param("features", numeric_features)
        
        if param_grid:
            mlflow.log_param("best_params", model_to_fit.best_params_)
            mlflow.log_metric("cv_mean_mse", -model_to_fit.best_score_)
            mlflow.log_metric("cv_std_mse", model_to_fit.cv_results_['std_test_score'].mean())
            
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        mlflow.sklearn.log_model(best_model, name="model", skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])
        
        # Сохраняем лучшую модель (E2) локально для дальнейших тестов и анализа
        if model_name == "E2_Ridge_Log":
            joblib.dump(best_model, "models/best_model.joblib")
            
        print(f"[{model_name}] MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        if param_grid:
            print(f"    Лучшие параметры: {model_to_fit.best_params_}")

# 6. Запуск экспериментов
if __name__ == "__main__":
    # E0 - Dummy (Среднее)
    run_experiment("E0_Dummy_Mean", DummyRegressor(strategy="mean"), base_numeric)
    
    # E1 - Linear (без GridSearch)
    run_experiment("E1_Linear_Log", LinearRegression(), base_numeric, use_log_target=True)
    
    # E2 - Ridge (с GridSearch по alpha и С ДОБАВЛЕНИЕМ объема)
    run_experiment("E2_Ridge_Log", Ridge(), extended_numeric, 
                   param_grid={"alpha": config["ridge"]["alphas"]}, 
                   use_log_target=True)
    
    # E3 - Random Forest (без объема, с GridSearch по глубине)
    run_experiment("E3_RandomForest", RandomForestRegressor(n_estimators=100, random_state=42), base_numeric, 
                   param_grid={"max_depth": [None, 3, 5, 10]})
    
    # E4 - Random Forest с объемом (Vproxy)
    run_experiment("E4_RandomForest_Volume", RandomForestRegressor(n_estimators=100, random_state=42), extended_numeric)
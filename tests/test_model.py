import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

# Загружаем данные и создаем признак (как в train.py)
df = pd.read_csv("fish_participant.csv")
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]

X = df.drop("Weight", axis=1)
y = df["Weight"]

# Разбиение (как в train.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Собираем пайплайн (Ridge с объемом)
numeric_features = ["Length1", "Length2", "Length3", "Height", "Width", "Vproxy"]
categorical_features = ["Species"]

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
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1
    ))
])

# Обучаем модель
model.fit(X_train, y_train)

print("Запуск тестов...")

# --- ТЕСТ 1: Индексы train и test не пересекаются ---
train_idx = set(X_train.index)
test_idx = set(X_test.index)
assert train_idx.isdisjoint(test_idx), "ОШИБКА: Индексы train и test пересекаются!"
print("Тест 1 (пересечение индексов): ПРОЙДЕН")

# --- ТЕСТ 2: Пайплайн принимает неизвестную категорию ---
# Создаем одну строку с новым видом рыбы ("NewFish"), которого нет в обучении
unseen_data = pd.DataFrame({
    "Species": ["NewFish"],
    "Length1": [20.0], "Length2": [22.0], "Length3": [25.0],
    "Height": [5.0], "Width": [3.0], "Vproxy": [25.0 * 5.0 * 3.0]
})
try:
    pred = model.predict(unseen_data)
    assert np.isfinite(pred[0]), "ОШИБКА: Прогноз не конечен!"
    print("Тест 2 (неизвестная категория): ПРОЙДЕН")
except Exception as e:
    raise AssertionError(f"ОШИБКА при предсказании для нового вида: {e}")

# --- ТЕСТ 3: Число прогнозов равно числу строк, все значения конечны ---
y_pred = model.predict(X_test)
assert len(y_pred) == len(X_test), "ОШИБКА: Количество прогнозов не совпадает с количеством строк!"
assert np.all(np.isfinite(y_pred)), "ОШИБКА: Среди прогнозов есть бесконечности или NaN!"
print("Тест 3 (размер и конечность прогнозов): ПРОЙДЕН")

# --- ТЕСТ 4: Метрика MSE на фиксированном примере (ручной расчет) ---
# Берем 2 первых объекта из теста
y_true_sample = y_test.iloc[:2].values
y_pred_sample = y_pred[:2]
manual_mse = np.mean((y_true_sample - y_pred_sample) ** 2)
sklearn_mse = mean_squared_error(y_true_sample, y_pred_sample)
assert np.isclose(manual_mse, sklearn_mse), "ОШИБКА: Ручной расчет MSE не совпадает со sklearn!"
print("Тест 4 (ручной расчет метрики): ПРОЙДЕН")

print("\nВсе тесты успешно пройдены!")
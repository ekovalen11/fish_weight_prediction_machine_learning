import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

# 1. Загружаем данные (точно так же, как в train.py)
df = pd.read_csv("fish_participant.csv")
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]

X = df.drop("Weight", axis=1)
y = df["Weight"]

# Используем то же разбиение, что и в train.py (тот же random_state!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Загружаем лучшую модель (Ridge с объемом, которая показала R2=0.96)
# Примечание: в реальном проекте мы бы сохранили её через joblib, 
# но сейчас мы просто обучим её заново для простоты (или загрузим из MLflow).
# Для этого скрипта мы обучим лучшую модель заново, чтобы не зависеть от MLflow.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

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

# Берем Ridge с логарифмированием (E2)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Расчет ошибок и остатков
errors = y_test - y_pred
absolute_errors = np.abs(errors)

# Создаем DataFrame для анализа
results_df = X_test.copy()
results_df["True_Weight"] = y_test
results_df["Predicted_Weight"] = y_pred
results_df["Absolute_Error"] = absolute_errors

# 4. Построение графика остатков
import os
os.makedirs("reports/figures", exist_ok=True) # Создаем папку для сохранения картинок

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('График остатков: Предсказанный вес vs Ошибка')
plt.xlabel('Предсказанный вес (г)')
plt.ylabel('Остаток (Истина - Прогноз)')
plt.tight_layout()
plt.savefig("reports/figures/residuals_plot.png")
plt.show()

# 5. Вывод 5 крупнейших ошибок
print("\n--- Топ-5 крупнейших ошибок ---")
top_5_errors = results_df.sort_values(by="Absolute_Error", ascending=False).head(5)
# Выводим только ключевые колонки, чтобы не было слишком широко
print(top_5_errors[["Species", "Length3", "Vproxy", "True_Weight", "Predicted_Weight", "Absolute_Error"]])

# 6. Сравнение MAE по видам
print("\n--- MAE по видам рыб ---")
mae_by_species = results_df.groupby("Species")["Absolute_Error"].mean().sort_values(ascending=False)
print(mae_by_species)

# 7. Проверка на физически невозможные предсказания (отрицательный вес)
negative_preds = (y_pred < 0).sum()
print(f"\nКоличество отрицательных предсказаний (невозможных): {negative_preds}")
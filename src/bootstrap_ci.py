import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error

# 1. Загружаем данные и делаем признак
df = pd.read_csv("fish_participant.csv")
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]
X = df.drop("Weight", axis=1)
y = df["Weight"]

# 2. Разбиение (как в train.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Собираем пайплайн (Ridge + объем)
numeric_features = ["Length1", "Length2", "Length3", "Height", "Width", "Vproxy"]
categorical_features = ["Species"]

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Реализация Bootstrap (1000 итераций)
n_iterations = 1000
n_size = len(y_test)
mae_scores = []

np.random.seed(42) # Фиксируем seed для воспроизводимости

print("Запуск бутстрэпа (1000 итераций)...")
for i in range(n_iterations):
    # Генерируем случайные индексы с возвращением
    indices = np.random.choice(n_size, size=n_size, replace=True)
    
    # Берем истинные значения и прогнозы по этим индексам
    y_true_sample = y_test.iloc[indices].values
    y_pred_sample = y_pred[indices]
    
    # Считаем MAE для этой выборки
    mae = mean_absolute_error(y_true_sample, y_pred_sample)
    mae_scores.append(mae)

# 5. Считаем доверительный интервал (95%)
lower_bound = np.percentile(mae_scores, 2.5)
upper_bound = np.percentile(mae_scores, 97.5)
mean_mae = np.mean(mae_scores)

print(f"\nРезультаты Bootstrap:")
print(f"Средний MAE: {mean_mae:.2f} г")
print(f"95% Доверительный интервал для MAE: [{lower_bound:.2f}; {upper_bound:.2f}] г")

# 6. Строим график распределения
plt.figure(figsize=(10, 5))
plt.hist(mae_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(lower_bound, color='red', linestyle='--', label=f'Нижняя граница 2.5% ({lower_bound:.2f})')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'Верхняя граница 97.5% ({upper_bound:.2f})')
plt.title('Распределение MAE при Bootstrap (1000 итераций)')
plt.xlabel('MAE (г)')
plt.ylabel('Частота')
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/bootstrap_mae_distribution.png")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Загружаем данные (точно так же, как в train.py)
df = pd.read_csv("fish_participant.csv")
df["Vproxy"] = df["Length3"] * df["Height"] * df["Width"]

X = df.drop("Weight", axis=1)
y = df["Weight"]

# Используем то же разбиение, что и в train.py (тот же random_state!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Загружаем лучшую сохраненную модель (Ridge с объемом)
model = joblib.load("models/best_model.joblib")

# Делаем предсказания
y_pred = model.predict(X_test)

# 3. Расчет ошибок и остатков
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

errors = y_test - y_pred
absolute_errors = np.abs(errors)

# Создаем DataFrame для анализа
results_df = X_test.copy()
results_df["True_Weight"] = y_test
results_df["Predicted_Weight"] = y_pred
results_df["Absolute_Error"] = absolute_errors

# Создаем папку для сохранения картинок, если ее нет
os.makedirs("reports/figures", exist_ok=True)

# 4. Построение графика остатков
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('График остатков: Предсказанный вес vs Ошибка')
plt.xlabel('Предсказанный вес (г)')
plt.ylabel('Остаток (Истина - Прогноз)')
plt.tight_layout()
plt.savefig("reports/figures/residuals_plot.png")
plt.close() # Закрываем график, чтобы он не висел в памяти

# 5. Вывод 5 крупнейших ошибок и сохранение в CSV
print("\n--- Топ-5 крупнейших ошибок ---")
top_5_errors = results_df.sort_values(by="Absolute_Error", ascending=False).head(5)
print(top_5_errors[["Species", "Length3", "Vproxy", "True_Weight", "Predicted_Weight", "Absolute_Error"]])
top_5_errors.to_csv("reports/top_5_errors.csv", index=False)

# 6. Сравнение MAE по видам
print("\n--- MAE по видам рыб ---")
mae_by_species = results_df.groupby("Species")["Absolute_Error"].mean().sort_values(ascending=False)
print(mae_by_species)

# 7. Сравнение MAE по диапазонам веса
print("\n--- MAE по диапазонам веса ---")
results_df["Size_Range"] = pd.cut(results_df["True_Weight"], bins=[0, 500, 2000], labels=["Мелкие (<500г)", "Крупные (>500г)"])
mae_by_range = results_df.groupby("Size_Range", observed=True)["Absolute_Error"].mean()
print(mae_by_range)

# 8. Проверка на физически невозможные предсказания
negative_preds = (y_pred < 0).sum()
print(f"\nКоличество отрицательных предсказаний (невозможных): {negative_preds}")

# 9. Логирование в MLflow (Дополнение к Разделу 5)
mlflow.set_experiment("Error_Analysis")
with mlflow.start_run(run_name="evaluate_best_model"):
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_artifact("reports/figures/residuals_plot.png")
    mlflow.log_artifact("reports/top_5_errors.csv")
    print("\n[MLflow] График остатков и таблица топ-5 ошибок успешно залогированы!")
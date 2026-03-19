import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("Admission_Predict.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["Serial No."])

X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_test_pred = model.predict(X_test_poly)

print(f"Точность (R²): {model.score(X_test_poly, y_test):.4f}")

joblib.dump(model, "admission_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(poly, "poly.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Модель и предобработка сохранены!")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='тестовые данные')
plt.plot([y_test.min(), y_test.max()],
         [y_test_pred.min(), y_test_pred.max()],
         color='red', linestyle='--', label='Идеальное предсказание')

plt.xlabel('Реальный шанс приема')
plt.ylabel('Предсказанный шанс приема')
plt.title('график')

plt.legend()
plt.grid(True)
plt.xlim(0.4, 1.0)
plt.ylim(0.4, 1.0)

plt.show()

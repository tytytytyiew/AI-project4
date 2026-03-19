APP.PY
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

st.title("Прогноз шансов поступления в университет")
df = pd.read_csv("Admission_Predict.csv")
df.columns = df.columns.str.strip()

st.header("Линейная Регресия")
st.header("Данные")

st.write("10 примеров из набора данных:")
st.dataframe(df.head(10))

if "Serial No." in df.columns:
    df = df.drop(columns=["Serial No."])
    st.write("Столбец 'Serial No.' удалён,")
st.subheader("Итоговый набор данных")
st.dataframe(df.head())
feature_cols = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA"]

X = df[feature_cols]
y = df["Chance of Admit"]

model_file = "admission_model.pkl"

if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_file)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.header("Метрики модели")

st.write("MSE (среднеквадратичная ошибка):", mse)
st.write("MAE (средняя абсолютная ошибка):", mae)
st.write("R2 (точность модели):", r2)

st.header("График: реальные значения vs прогноз")

fig, ax = plt.subplots()
ax.scatter(y, y_pred)
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', alpha=0.5, linewidth=2, label='Идеальная линия')


ax.set_xlabel("Реальные значения")
ax.set_ylabel("Предсказанные значения")
ax.set_title("Сравнение реальных и предсказанных значений")

st.pyplot(fig)
st.header("Введите данные для прогноза")
GRE = st.slider("GRE Score", int(df["GRE Score"].min()), int(df["GRE Score"].max()), int(df["GRE Score"].mean()))
TOEFL = st.slider("TOEFL Score", int(df["TOEFL Score"].min()), int(df["TOEFL Score"].max()), int(df["TOEFL Score"].mean()))
UniRating = st.slider("University Rating", int(df["University Rating"].min()), int(df["University Rating"].max()), int(df["University Rating"].mean()))
SOP = st.slider("SOP", float(df["SOP"].min()), float(df["SOP"].max()), float(df["SOP"].mean()))
LOR = st.slider("LOR", float(df["LOR"].min()), float(df["LOR"].max()), float(df["LOR"].mean()))
CGPA = st.slider("CGPA", float(df["CGPA"].min()), float(df["CGPA"].max()), float(df["CGPA"].mean()))

input_data = pd.DataFrame({
    "GRE Score": [GRE],
    "TOEFL Score": [TOEFL],
    "University Rating": [UniRating],
    "SOP": [SOP],
    "LOR": [LOR],
    "CGPA": [CGPA]
})

if st.button("Прогнозировать"):
    prediction = model.predict(input_data)[0]

    st.subheader("Результат прогноза")

    st.success(f"Шанс поступления: {prediction*100:.2f}%")

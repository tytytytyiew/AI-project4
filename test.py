import pandas as pd
import joblib

def get_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                print(f"Введите число от {min_val} до {max_val}")
                continue
            return value
        except ValueError:
            print("Ошибка! Введите число.")

model = joblib.load("admission_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")
columns = joblib.load("columns.pkl")

gre = get_float("GRE Score (0-340): ", 0, 340)
toefl = get_float("TOEFL Score (0-120): ", 0, 120)
university = get_float("University Rating (1-5): ", 1, 5)
sop = get_float("SOP (1-5): ", 1, 5)
lor = get_float("LOR (1-5): ", 1, 5)
cgpa = get_float("CGPA (0-10): ", 0, 10)
research = get_float("Research (0 или 1): ", 0, 1)

student_df = pd.DataFrame([[gre, toefl, university, sop, lor, cgpa, research]], columns=columns)

student_scaled = scaler.transform(student_df)
student_poly = poly.transform(student_scaled)

prediction = model.predict(student_poly)
prediction = max(0, min(1, prediction[0]))

print(f"\nШанс поступления: {prediction*100:.2f}%")

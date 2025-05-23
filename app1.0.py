from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import sqlite3

app = Flask(__name__)

# 1. Load model & encoders at startup
MODEL_PATH = 'model_le/model.pkl'
ENCODERS = {
    'sex':                   'model_le/le_sex.pkl',
    'chest_pain_type':       'model_le/le_chest_pain_type.pkl',
    'fasting_blood_sugar':   'model_le/le_fasting_blood_sugar.pkl',
    'heart_ecg':             'model_le/le_heart_ecg.pkl',
    'exercise_induced_angina': 'model_le/le_exercise_induced_angina.pkl',
}

model = joblib.load(MODEL_PATH)
label_encoders = {name: joblib.load(path)
                  for name, path in ENCODERS.items()}


@app.route('/', methods=['GET'])
def form():
    # Render the HTML form
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 2. Pull values from form
    # Categorical:
    sex       = request.form['sex']
    cp_type   = request.form['chest_pain_type']
    fbs       = request.form['fasting_blood_sugar']
    ecg       = request.form['heart_ecg']
    angina    = request.form['exercise_induced_angina']

    # Numeric (convert to float/int as needed):
    age       = float(request.form['age'])
    trestbps  = float(request.form['trestbps'])
    chol      = float(request.form['chol'])
    thalach   = float(request.form['thalach'])
    oldpeak   = float(request.form['oldpeak'])

    # 3. Encode categoricals
    sex_num    = label_encoders['sex'].transform([sex])[0]
    cp_num     = label_encoders['chest_pain_type'].transform([cp_type])[0]
    fbs_num    = label_encoders['fasting_blood_sugar'].transform([fbs])[0]
    ecg_num    = label_encoders['heart_ecg'].transform([ecg])[0]
    angina_num = label_encoders['exercise_induced_angina'].transform([angina])[0]

    # 4. Build feature vector in the same order your model expects
    features = [
        age, trestbps, chol, thalach, oldpeak,
        sex_num, cp_num, fbs_num, ecg_num, angina_num,
    ]
    X_input = np.array(features).reshape(1, -1)

    # 5. Predict
    pred_prob = model.predict_proba(X_input)
    max_index = np.argmax(pred_prob[0])  # Get index of highest probability

    label_map = {
        0: 'No heart disease',
        1: 'heart disease stage 1',
        2: 'heart disease stage 2',
        3: 'heart disease stage 3',
        4: 'heart disease stage 4'
    }

    # Get both label and probability using the max index
    result_label = label_map[max_index]
    max_probability = round(pred_prob[0, max_index], 3) * 100


    labels = list(label_map.values())  # Get ordered list of class names
    prob_percentages = [round(prob * 100, 1) for prob in pred_prob[0]]  # Convert to percentages

    # 6. Render result page
    return render_template(
        'results.html',
        prediction=result_label,
        max_probability=f"{max_probability}",
        all_probabilities=prob_percentages,
        class_labels=labels,
        zip = zip
    )

if __name__ == '__main__':
    os.makedirs('model_le', exist_ok=True)
    app.run(debug=True)

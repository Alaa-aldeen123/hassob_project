from flask import Flask, render_template, request, g
import joblib
import os
import numpy as np
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

# ─── Model & Encoder setup ────────────────────────────────────────────────────
MODEL_PATH = 'model_le/model.pkl'
ENCODERS = {
    'sex':                   'model_le/le_sex.pkl',
    'chest_pain_type':       'model_le/le_chest_pain_type.pkl',
    'fasting_blood_sugar':   'model_le/le_fasting_blood_sugar.pkl',
    'heart_ecg':             'model_le/le_heart_ecg.pkl',
    'exercise_induced_angina': 'model_le/le_exercise_induced_angina.pkl',
}

model = joblib.load(MODEL_PATH)
label_encoders = {name: joblib.load(path) for name, path in ENCODERS.items()}

# ─── SQLite helper functions ──────────────────────────────────────────────────
DATABASE = os.path.join(os.path.dirname(__file__), 'litsql.db')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                age REAL, sex TEXT, chest_pain_type TEXT,
                fasting_blood_sugar TEXT, heart_ecg TEXT,
                exercise_induced_angina TEXT,
                resting_blood_pressure REAL, cholesterol REAL,
                maximum_heart_rate REAL, oldpeak REAL,
                prediction TEXT,
                probabilities TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

# Initialize database at startup
init_db()

# ─── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Read inputs
    patient_name = request.form['patient_name']
    sex       = request.form['sex']
    cp_type   = request.form['chest_pain_type']
    fbs       = request.form['fasting_blood_sugar']
    ecg       = request.form['heart_ecg']
    angina    = request.form['exercise_induced_angina']
    age       = float(request.form['age'])
    trestbps  = float(request.form['trestbps'])
    chol      = float(request.form['chol'])
    thalach   = float(request.form['thalach'])
    oldpeak   = float(request.form['oldpeak'])

    # 2. Encode categoricals
    sex_num    = label_encoders['sex'].transform([sex])[0]
    cp_num     = label_encoders['chest_pain_type'].transform([cp_type])[0]
    fbs_num    = label_encoders['fasting_blood_sugar'].transform([fbs])[0]
    ecg_num    = label_encoders['heart_ecg'].transform([ecg])[0]
    angina_num = label_encoders['exercise_induced_angina'].transform([angina])[0]

    # 3. Predict
    features = [age, trestbps, chol, thalach, oldpeak,
                sex_num, cp_num, fbs_num, ecg_num, angina_num]
    X_input = np.array(features).reshape(1, -1)
    pred_prob = model.predict_proba(X_input)[0]

    # 4. Map to labels
    label_map = {
        0: 'No heart disease',
        1: 'heart disease stage 1',
        2: 'heart disease stage 2',
        3: 'heart disease stage 3',
        4: 'heart disease stage 4'
    }
    max_index = int(np.argmax(pred_prob))
    result_label = label_map[max_index]
    prob_percentages = [round(p * 100, 1) for p in pred_prob]

    # 5. Save to SQLite
    db = get_db()
    db.execute('''
        INSERT INTO predictions (
            patient_name, age, sex, chest_pain_type,
            fasting_blood_sugar, heart_ecg,
            exercise_induced_angina,
            resting_blood_pressure, cholesterol,
            maximum_heart_rate, oldpeak,
            prediction, probabilities
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_name, age, sex, cp_type,
        fbs, ecg, angina,
        trestbps, chol, thalach, oldpeak,
        result_label, json.dumps(prob_percentages)
    ))
    db.commit()

    return render_template(
        'results.html',
        prediction=result_label,
        class_labels=list(label_map.values()),
        all_probabilities=prob_percentages
    )


@app.route('/history', methods=['GET'])
def history():
    db = get_db()
    cur = db.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cur.fetchall()
    # deserialize probabilities
    history = []
    for r in rows:
        history.append({
            'patient_name':    r['patient_name'],
            'timestamp':       r['timestamp'],
            'inputs': {
                'age': r['age'],
                'sex': r['sex'],
                'chest_pain_type': r['chest_pain_type'],
                'fasting_blood_sugar': r['fasting_blood_sugar'],
                'heart_ecg': r['heart_ecg'],
                'exercise_induced_angina': r['exercise_induced_angina'],
                'resting_blood_pressure': r['resting_blood_pressure'],
                'cholesterol': r['cholesterol'],
                'maximum_heart_rate': r['maximum_heart_rate'],
                'oldpeak': r['oldpeak'],
            },
            'prediction':      r['prediction'],
            'probabilities':   json.loads(r['probabilities'])
        })
    return render_template('history.html', history=history)


if __name__ == '__main__':
    os.makedirs('model_le', exist_ok=True)
    app.run(debug=True)

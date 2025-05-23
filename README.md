# hassob_project
The final project for AI course to Hassob Academy

# Heart Disease Prediction Web App

A Flask-based web application that predicts the stage of heart disease based on patient data using a trained machine learning model and stores prediction history in a SQLite database.

## Table of Contents

* [Features](#features)
* [Demo](#demo)
* [Technology Stack](#technology-stack)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Folder Structure](#folder-structure)
* [Usage](#usage)
* [API Endpoints](#api-endpoints)
* [Database](#database)
* [Contributing](#contributing)
* [License](#license)

## Features

* Input patient details via a web form
* Preprocess and encode categorical variables
* Predict heart disease stage (0–4) with probability scores
* Store predictions and input data in SQLite
* View past predictions in a history page

## Demo

1. Open the home page (`/`) to enter patient information.
2. Submit the form to view prediction results and probabilities.
3. Navigate to `/history` to review saved predictions.

## Technology Stack

* **Backend:** Flask 3.1.1
* **Machine Learning:** scikit-learn 1.5.1, joblib 1.4.2, numpy 2.2.6
* **Database:** SQLite
* **Templates:** Jinja2 (included with Flask)

## Prerequisites

* Python 3.8 or higher
* Git

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Alaa-aldeen123/Hassob_Course/Hassob_project.git
   cd Hassob_project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model and encoders are in place**

   * `model_le/model.pkl` — trained ML model
   * `model_le/le_*.pkl` — label encoders for categorical features

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Access the app** Open your browser and navigate to `http://127.0.0.1:5000/`.

## Folder Structure

```
heart-disease-prediction/
├── app.py                 # Flask application entry point
├── requirements.txt       # Python package dependencies
├── litsql.db              # SQLite database (auto-generated)
├── model_le/              # Trained model and label encoders
│   ├── model.pkl
│   ├── le_sex.pkl
│   ├── le_chest_pain_type.pkl
│   ├── le_fasting_blood_sugar.pkl
│   ├── le_heart_ecg.pkl
│   └── le_exercise_induced_angina.pkl
└── templates/             # HTML templates
    ├── form.html         # Input form page
    ├── results.html      # Prediction results page
    └── history.html      # Prediction history page
```

## Usage

1. **Home Page** (`/`): Enter patient name, age, categorical inputs (sex, chest pain type, etc.), and numeric vitals.
2. **Predict** (`/predict`): View the predicted heart disease stage and probability for each class.
3. **History** (`/history`): Browse previous predictions with timestamps and inputs.

## API Endpoints

| Route      | Method | Description                                  |
| ---------- | ------ | -------------------------------------------- |
| `/`        | GET    | Render patient input form                    |
| `/predict` | POST   | Process form, predict, save, and show result |
| `/history` | GET    | Retrieve and display past predictions        |

## Database

* **File:** `litsql.db`
* **Table:** `predictions`

  * `id`, `patient_name`, `age`, `sex`, `chest_pain_type`, `fasting_blood_sugar`, `heart_ecg`, `exercise_induced_angina`, `resting_blood_pressure`, `cholesterol`, `maximum_heart_rate`, `oldpeak`, `prediction`, `probabilities`, `timestamp`
* Initialized automatically on app startup.

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import shap
import json
import mysql.connector


db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",    
    database="loan_system"
)

cursor = db.cursor()

# Load trained model & metadata

model = tf.keras.models.load_model("loan_model.keras")

with open("data_columns.json", "r") as file:
    columns = json.load(file)["data_columns"]

# Initialize Flask app
app = Flask(__name__)


background_data = np.zeros((1, len(columns)))  
explainer = shap.KernelExplainer(model.predict, background_data)


# Prediction and SHAP Implementation

def predict_decision(emp_length, grade, home_ownership, purpose, term, annual_income, installment, loan_amount):
    X = np.zeros(len(columns))

    # Map categorical features
    cat_features = {
        "Emp_length": emp_length,
        "Grade": grade,
        "Home_ownership": home_ownership,
        "Purpose": purpose,
        "Term": str(term)
    }

    for val in cat_features.values():
        if val in columns:
            X[columns.index(val)] = 1

    # Map numerical features
    for num_col, num_val in zip(["annual_income", "installment", "loan_amount"],
                                [annual_income, installment, loan_amount]):
        if num_col in columns:
            X[columns.index(num_col)] = num_val

    # Predict
    pred = model.predict(X.reshape(1, -1))[0][0]

    # SHAP explanation for this input
    shap_values = explainer.shap_values(X.reshape(1, -1))
    shap_importance = sorted(
        list(zip(columns, shap_values[0])), key=lambda x: abs(x[1]), reverse=True
    )[:3]

    top_features = [
        f"{feat} ({'increased' if val > 0 else 'decreased'} approval likelihood)"
        for feat, val in shap_importance
    ]

    return pred, top_features


# Routes

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        emp_length = request.form["emp_length"]
        grade = request.form["grade"]
        home_ownership = request.form["home_ownership"]
        purpose = request.form["purpose"]
        term = request.form["term"]
        annual_income = float(request.form["annual_income"])
        loan_amount = float(request.form["loan_amount"])
        installment = float(request.form["installment"])

        # ML prediction
        prediction, top_features = predict_decision(
            emp_length, grade, home_ownership, purpose, term,
            annual_income, installment, loan_amount
        )

        decision = "Loan Approved" if prediction >= 0.5 else "Loan Likely to Default"
        probability = round(float(prediction) * 100, 2)

        # SAVE TO DATABASE 
        sql = """
            INSERT INTO loan_predictions
            (emp_length, grade, home_ownership, purpose, term, annual_income,
             loan_amount, installment, decision, probability)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            emp_length, grade, home_ownership, purpose, term,
            annual_income, loan_amount, installment,
            decision, probability
        )

        cursor.execute(sql, values)
        db.commit()

      
        return jsonify({
            "decision": decision,
            "probability": probability,
            "explanation": ", ".join(top_features) 
        })

    except Exception as e:
        return jsonify({"decision": f"Error: {e}", "probability": 0, "explanation": ""})



# Run app
if __name__ == "__main__":
    app.run(debug=True)

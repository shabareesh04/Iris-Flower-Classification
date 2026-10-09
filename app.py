"""
Iris Flower Classification - Flask Web Application.

Provides a clean web interface and REST prediction endpoint:
- Loads the pre-trained ML model and metadata at startup (no retraining).
- Performs robust server-side validation against non-numeric, negative, or unrealistic values.
- Returns predicted species, confidence scores (if supported), and input summaries.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.json")

# Species dictionary with botanical insights
SPECIES_INFO = {
    "Iris Setosa": {
        "icon": "🌸",
        "description": "Characterized by distinctly small, narrow petals and wide sepals. Linearly separable from other Iris species.",
        "badge_class": "badge-setosa",
    },
    "Iris Versicolor": {
        "icon": "🌺",
        "description": "Medium-sized petals and sepals. Represents an intermediate botanical profile between Setosa and Virginica.",
        "badge_class": "badge-versicolor",
    },
    "Iris Virginica": {
        "icon": "🌼",
        "description": "Known for the largest petal lengths and widths. Often exhibits deep violet-blue blossoms.",
        "badge_class": "badge-virginica",
    },
}

# Botanical sanity check boundaries (in cm)
REALISTIC_BOUNDS = {
    "sepal_length": (3.0, 12.0),
    "sepal_width": (1.5, 6.0),
    "petal_length": (0.5, 10.0),
    "petal_width": (0.1, 4.0),
}


def load_model_and_metadata():
    """Load serialized model and JSON metadata if available."""
    model = None
    info = {
        "model_name": "Trained Classifier",
        "accuracy": 0.0,
        "class_names": ["Iris Setosa", "Iris Versicolor", "Iris Virginica"],
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    }

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            app.logger.error(f"Error loading model from {MODEL_PATH}: {e}")

    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading metadata from {INFO_PATH}: {e}")

    return model, info


model, model_info = load_model_and_metadata()


def validate_input(data):
    """
    Validate input dictionary against:
    - Missing keys
    - Empty strings
    - Non-numeric formats
    - Negative or zero values
    - Unrealistic botanical ranges
    """
    fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    cleaned = {}

    for field in fields:
        raw_val = data.get(field)
        if raw_val is None or str(raw_val).strip() == "":
            return None, f"Measurement '{field.replace('_', ' ').title()}' is required."

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return None, f"'{field.replace('_', ' ').title()}' must be a valid numeric number."

        if val <= 0:
            return None, f"'{field.replace('_', ' ').title()}' must be greater than zero."

        min_val, max_val = REALISTIC_BOUNDS[field]
        if val < min_val or val > max_val:
            return (
                None,
                f"'{field.replace('_', ' ').title()}' value ({val} cm) is biologically unrealistic. Realistic range is {min_val} cm to {max_val} cm.",
            )

        cleaned[field] = val

    return cleaned, None


@app.route("/", methods=["GET"])
def home():
    """Render prediction homepage."""
    return render_template(
        "index.html",
        model_info=model_info,
        bounds=REALISTIC_BOUNDS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request and return results or clear validation error."""
    if model is None:
        error_msg = "Model artifact not found. Please run 'python train.py' to generate the model."
        if request.is_json:
            return jsonify({"status": "error", "message": error_msg}), 500
        return render_template("index.html", error=error_msg, model_info=model_info), 500

    # Extract data from JSON or Form submission
    if request.is_json:
        data = request.get_json() or {}
    else:
        data = request.form.to_dict()

    cleaned_data, error_msg = validate_input(data)
    if error_msg:
        if request.is_json:
            return jsonify({"status": "error", "message": error_msg}), 400
        return render_template(
            "index.html",
            error=error_msg,
            form_data=data,
            model_info=model_info,
            bounds=REALISTIC_BOUNDS,
        ), 400

    # Prepare features in the exact expected feature order with column names
    feature_names = model_info.get(
        "feature_names", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    features = pd.DataFrame([[
        cleaned_data["sepal_length"],
        cleaned_data["sepal_width"],
        cleaned_data["petal_length"],
        cleaned_data["petal_width"],
    ]], columns=feature_names)

    # Make class prediction
    pred_idx = int(model.predict(features)[0])
    class_names = model_info.get(
        "class_names", ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
    )
    predicted_species = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"

    # Prediction Confidence (only if model natively supports predict_proba)
    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(features)[0]
            confidence_pct = round(float(proba_arr[pred_idx]) * 100, 2)
            confidence = f"{confidence_pct:.2f}%"

            probabilities = {
                class_names[i]: round(float(proba_arr[i]) * 100, 2)
                for i in range(len(class_names))
            }
        except Exception:
            confidence = None
            probabilities = None

    species_details = SPECIES_INFO.get(
        predicted_species,
        {"icon": "🌿", "description": "Iris flower specimen.", "badge_class": "badge-default"},
    )

    result_payload = {
        "status": "success",
        "species": predicted_species,
        "icon": species_details["icon"],
        "description": species_details["description"],
        "confidence": confidence,
        "probabilities": probabilities,
        "measurements": cleaned_data,
        "model_name": model_info.get("model_name", "Trained Model"),
        "model_accuracy": model_info.get("accuracy", None),
    }

    if request.is_json:
        return jsonify(result_payload)

    return render_template(
        "result.html",
        result=result_payload,
        model_info=model_info,
    )


@app.route("/api/info", methods=["GET"])
def api_info():
    """Return model metadata in JSON format."""
    return jsonify({
        "status": "success",
        "model_info": model_info,
        "bounds": REALISTIC_BOUNDS,
    })


if __name__ == "__main__":
    # Host on 0.0.0.0 or localhost on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)

"""
Unit tests for the trained Iris Machine Learning model artifact.
"""

import os
import json
import pytest
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.json")

VALID_CLASSES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


def test_model_file_exists():
    """Verify that the model file exists on disk."""
    assert os.path.exists(MODEL_PATH), f"Model binary not found at {MODEL_PATH}"


def test_model_info_exists():
    """Verify that the model metadata JSON exists and contains expected keys."""
    assert os.path.exists(INFO_PATH), f"Model info not found at {INFO_PATH}"
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    assert "model_name" in info
    assert "accuracy" in info
    assert "feature_names" in info
    assert "class_names" in info
    assert len(info["feature_names"]) == 4
    assert len(info["class_names"]) == 3


import pandas as pd


def test_model_can_be_loaded():
    """Verify that joblib successfully deserializes the model."""
    model = joblib.load(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict"), "Loaded object does not implement predict()"


def test_model_accepts_four_features():
    """Verify that the model accepts an array with 4 numeric features."""
    model = joblib.load(MODEL_PATH)
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    sample_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=info["feature_names"])
    pred = model.predict(sample_input)
    assert len(pred) == 1


def test_model_returns_valid_class():
    """Verify that model output maps to one of the 3 designated Iris species."""
    model = joblib.load(MODEL_PATH)
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    class_names = info["class_names"]
    feature_names = info["feature_names"]

    # Test distinct samples across the feature spectrum
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # Typical Setosa
        [5.9, 3.0, 4.2, 1.5],  # Typical Versicolor
        [6.5, 3.0, 5.5, 1.8],  # Typical Virginica
    ]

    for sample in samples:
        input_df = pd.DataFrame([sample], columns=feature_names)
        pred_idx = int(model.predict(input_df)[0])
        assert pred_idx in [0, 1, 2], f"Predicted class index {pred_idx} out of range [0, 2]"
        predicted_species = class_names[pred_idx]
        assert predicted_species in VALID_CLASSES

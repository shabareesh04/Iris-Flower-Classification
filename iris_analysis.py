"""
Iris Flower Classification - Exploratory Data Analysis & Modeling Notebook Script.

This standalone exploratory script demonstrates:
- Comprehensive statistical data inspection
- Multidimensional visualization analysis
- Feature separability and correlation assessment
- Comparative model benchmarking across 5 algorithms
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def run_analysis():
    print("=" * 60)
    print("IRIS FLOWER DATASET - IN-DEPTH EXPLORATORY ANALYSIS")
    print("=" * 60)

    # 1. Load Dataset
    iris = load_iris()
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(iris.data, columns=feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(lambda x: iris.target_names[x])

    print("\n[1] Basic Characteristics:")
    print(f"Total Observations: {df.shape[0]}")
    print(f"Feature Dimensions: {df.shape[1] - 2} features")
    print(f"Species Classes: {list(iris.target_names)}")

    print("\n[2] Summary Statistics per Species:")
    grouped_stats = df.groupby("species")[feature_names].mean()
    print(grouped_stats.round(2))

    print("\n[3] Inter-Feature Correlations:")
    corr = df[feature_names].corr()
    print(corr.round(3))
    print("\nInsight: Petal length and petal width show an extremely strong positive")
    print("correlation (r > 0.96), making them premier discriminative features.")

    # 4. Benchmarking
    X = df[feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
    }

    print("\n[4] Benchmark Results (Test Accuracy):")
    print("-" * 50)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name:<28}: {acc * 100:.2f}%")

    print("-" * 50)
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    run_analysis()

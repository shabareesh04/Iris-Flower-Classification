"""
Iris Flower Classification - Model Training and Evaluation Pipeline.

This script executes the complete machine learning workflow:
1. Loads the Iris dataset from Scikit-Learn
2. Performs data inspection, preprocessing, and quality validation
3. Generates and saves Exploratory Data Analysis (EDA) visualizations
4. Splits the dataset into stratified training and testing sets (80/20)
5. Trains and evaluates 5 classification algorithms
6. Evaluates metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
7. Automatically selects and saves the top-performing model and metadata
"""

import sys
import os
import json
from datetime import datetime

# Ensure standard output uses UTF-8 or safe encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for scripts and servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Paths setup (relative to script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Standard feature and class labels
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


def load_and_prepare_data():
    """Load the Iris dataset and convert it into a Pandas DataFrame."""
    print("\n[Step 1 & 2] Loading Iris dataset and preparing DataFrame...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_NAMES)
    df["target"] = iris.target
    return df, iris


def explore_dataset(df):
    """Print key exploration statistics for the dataset."""
    print("\n[Step 3] Dataset Exploration:")
    print("-" * 50)
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nStatistical Summary:")
    print(df.describe().round(3))
    print("\nClass Distribution:")
    class_counts = df["target"].value_counts().sort_index()
    for idx, count in class_counts.items():
        print(f"  Class {idx} ({CLASS_NAMES[idx]}): {count} samples")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("-" * 50)


def preprocess_data(df):
    """Verify data hygiene, duplicates, and missing values."""
    print("\n[Step 4] Data Preprocessing & Validation...")
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Total Missing Values: {missing_count}")
    print(f"Duplicate Rows Detected: {duplicate_count}")

    # Note: In biological datasets like Iris, identical physical measurements across
    # separate specimens are legitimate observations. No rows missing or corrupted.
    cleaned_df = df.copy()

    # Verify numerical ranges (non-negative and realistic dimensions)
    for col in FEATURE_NAMES:
        if (cleaned_df[col] <= 0).any():
            raise ValueError(f"Invalid non-positive value detected in column {col}")

    print("Data validation passed: All values are numeric, positive, and complete.")
    print(f"Dataset shape retained: {cleaned_df.shape}")
    return cleaned_df


def perform_eda(df):
    """Generate and export publication-quality EDA plots to the outputs folder."""
    print("\n[Step 5] Generating Exploratory Data Analysis (EDA) Visualizations...")
    sns.set_theme(style="whitegrid", palette="muted")
    plot_df = df.copy()
    plot_df["species"] = plot_df["target"].map(lambda x: CLASS_NAMES[x])

    # 1. Pair Plot
    pairplot = sns.pairplot(
        plot_df,
        vars=FEATURE_NAMES,
        hue="species",
        diag_kind="kde",
        markers=["o", "s", "D"],
        palette=["#3B82F6", "#10B981", "#8B5CF6"],
    )
    pairplot.fig.suptitle("Iris Feature Pair Plot by Species", y=1.02, fontsize=14, fontweight="bold")
    pairplot_path = os.path.join(OUTPUTS_DIR, "eda_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved pair plot: {pairplot_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[FEATURE_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    heatmap_path = os.path.join(OUTPUTS_DIR, "eda_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved correlation heatmap: {heatmap_path}")

    # 3. Sepal Length vs Sepal Width Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        style="species",
        s=85,
        palette=["#3B82F6", "#10B981", "#8B5CF6"],
    )
    plt.title("Sepal Length vs. Sepal Width", fontsize=13, fontweight="bold")
    plt.xlabel("Sepal Length (cm)", fontsize=11)
    plt.ylabel("Sepal Width (cm)", fontsize=11)
    plt.legend(title="Species")
    sepal_scatter_path = os.path.join(OUTPUTS_DIR, "eda_sepal_scatter.png")
    plt.savefig(sepal_scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved sepal scatter plot: {sepal_scatter_path}")

    # 4. Petal Length vs Petal Width Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="petal_length",
        y="petal_width",
        hue="species",
        style="species",
        s=85,
        palette=["#3B82F6", "#10B981", "#8B5CF6"],
    )
    plt.title("Petal Length vs. Petal Width", fontsize=13, fontweight="bold")
    plt.xlabel("Petal Length (cm)", fontsize=11)
    plt.ylabel("Petal Width (cm)", fontsize=11)
    plt.legend(title="Species")
    petal_scatter_path = os.path.join(OUTPUTS_DIR, "eda_petal_scatter.png")
    plt.savefig(petal_scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved petal scatter plot: {petal_scatter_path}")

    # 5. Class Distribution Plot
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=plot_df,
        x="species",
        hue="species",
        legend=False,
        palette=["#3B82F6", "#10B981", "#8B5CF6"],
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title("Iris Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Flower Species", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
    distribution_path = os.path.join(OUTPUTS_DIR, "eda_class_distribution.png")
    plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved class distribution plot: {distribution_path}")


def split_dataset(df):
    """Split data into stratified train and test sets (80/20)."""
    print("\n[Step 6] Performing Stratified Train/Test Split (80/20)...")
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def get_models():
    """Instantiate the 5 algorithms to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train each algorithm, compute performance metrics, and record details."""
    print("\n[Step 7 & 8] Training and Evaluating Models...")
    results = []
    trained_instances = {}
    reports = {}
    matrices = {}

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_instances[name] = model

        # Predict
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

        reports[name] = report
        matrices[name] = cm

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
        })

    # Save comparison dataframe to outputs/model_comparison.csv
    results_df = pd.DataFrame(results)
    comparison_csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"  [+] Model comparison exported to: {comparison_csv_path}")

    # Plot Confusion Matrix for all models
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, (name, cm) in enumerate(matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar=False,
        )
        axes[i].set_title(f"{name}\nAccuracy: {results_df.loc[results_df['Model'] == name, 'Accuracy'].values[0]*100:.2f}%")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # Remove extra subplot
    if len(matrices) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Combined confusion matrices exported to: {cm_plot_path}")

    return results_df, trained_instances, reports


def select_best_model(results_df, trained_instances):
    """
    Select the best-performing model based primarily on Accuracy,
    using F1-Score as tie-breaker.
    """
    sorted_df = results_df.sort_values(
        by=["Accuracy", "F1_Score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_name = best_row["Model"]
    best_model = trained_instances[best_name]

    return best_name, best_model, best_row


def save_model_artifacts(best_name, best_model, best_stats):
    """Save the serialized model with joblib and store model metadata in JSON."""
    print("\n[Step 9] Saving Best Model and Metadata...")
    model_file_path = os.path.join(MODEL_DIR, "iris_model.pkl")
    joblib.dump(best_model, model_file_path)
    print(f"  [+] Model binary saved: {model_file_path}")

    metadata = {
        "model_name": best_name,
        "accuracy": round(float(best_stats["Accuracy"]), 4),
        "precision": round(float(best_stats["Precision"]), 4),
        "recall": round(float(best_stats["Recall"]), 4),
        "f1_score": round(float(best_stats["F1_Score"]), 4),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = os.path.join(MODEL_DIR, "model_info.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  [+] Model metadata saved: {metadata_path}")


def main():
    """Execute complete ML pipeline."""
    # Step 1 & 2: Load and Prepare
    raw_df, _ = load_and_prepare_data()

    # Step 3: Explore
    explore_dataset(raw_df)

    # Step 4: Preprocess
    df = preprocess_data(raw_df)

    # Step 5: EDA Visualizations
    perform_eda(df)

    # Step 6: Split
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Step 7 & 8: Train & Evaluate
    models = get_models()
    results_df, trained_instances, reports = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # Step 9: Best Model Selection & Saving
    best_name, best_model, best_row = select_best_model(results_df, trained_instances)
    save_model_artifacts(best_name, best_model, best_row)

    # Final Summary Print matching required format
    print("\n" + "=" * 40)
    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 40)
    print(f"\nDataset Shape: {raw_df.shape}\n")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print(f"Number of Classes: {len(CLASS_NAMES)}\n")
    print("Model Performance")
    print("-" * 40)
    for _, row in results_df.iterrows():
        name_str = f"{row['Model']:<20}"
        acc_str = f"{row['Accuracy'] * 100:.2f}%"
        print(f"{name_str}: {acc_str}")
    print("\nBest Model:")
    print(f"{best_name}\n")
    print("Model saved successfully:")
    print("model/iris_model.pkl")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()

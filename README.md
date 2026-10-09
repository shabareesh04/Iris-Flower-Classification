# Iris Flower Classification Using Machine Learning

A complete, end-to-end Machine Learning web application and data pipeline designed to classify Iris flower specimens into three distinct botanical species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) based on sepal and petal physical dimensions.

---

## 1. Description

The **Iris Flower Classification System** demonstrates the end-to-end Machine Learning lifecycle, spanning:
1. Automated dataset acquisition from Scikit-learn
2. Data inspection, validation, and sanitization
3. Comprehensive Exploratory Data Analysis (EDA) with publication-quality visualizations
4. Comparative model benchmarking across five classification algorithms
5. Automated selection and serialization of the best-performing model
6. A responsive, modern web interface powered by Flask, HTML5, CSS3, and JavaScript
7. Comprehensive unit and integration test coverage

---

## 2. Key Features

- **End-to-End Workflow:** From raw biological measurements to real-time model inference.
- **5-Algorithm Benchmark:** Evaluates Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine.
- **Automated Best Model Selection:** Automatically ranks algorithms by test accuracy (using F1-Score as tie-breaker) and persists model binaries and metadata.
- **Rich Visualizations:** Generates pair plots, correlation heatmaps, class distributions, and confusion matrices saved to `outputs/`.
- **Production-Ready Web Application:** Lightweight Flask server with decoupled frontend.
- **Strict Input Validation:** Multi-layer validation (client-side and server-side) preventing invalid, non-numeric, negative, or biologically unrealistic inputs.
- **Modern Academic UI:** Responsive card-based layout featuring smooth micro-animations, quick-sample loaders, and probability breakdown charts.
- **Comprehensive Test Suite:** 13 unit and integration tests via Pytest validating data loading, model behavior, and web endpoints.

---

## 3. Dataset

The project utilizes the classic **Fisher's Iris Flower Dataset** bundled directly within Scikit-learn (`sklearn.datasets.load_iris`).

- **Total Samples:** 150
- **Balanced Distribution:** 50 observations per class
- **Number of Features:** 4 continuous numeric measurements (measured in centimeters)
- **Target Classes:** 3 species:
  - `0` &rarr; **Iris Setosa**
  - `1` &rarr; **Iris Versicolor**
  - `2` &rarr; **Iris Virginica**

---

## 4. Input Features

The classifier uses four anatomical dimensions:

| Feature Name | Description | Typical Botanical Range |
| :--- | :--- | :--- |
| **Sepal Length** | Length of the outer flower protective leaf | `4.3 cm – 7.9 cm` (accepted: `3.0 – 12.0 cm`) |
| **Sepal Width** | Width of the outer flower protective leaf | `2.0 cm – 4.4 cm` (accepted: `1.5 – 6.0 cm`) |
| **Petal Length** | Length of the inner blossom petal | `1.0 cm – 6.9 cm` (accepted: `0.5 – 10.0 cm`) |
| **Petal Width** | Width of the inner blossom petal | `0.1 cm – 2.5 cm` (accepted: `0.1 – 4.0 cm`) |

---

## 5. Machine Learning Algorithms

The pipeline trains and benchmarks five foundational supervised classification algorithms:

1. **Logistic Regression:**
   - Linear probabilistic classifier using multinomial cross-entropy loss.
   - Provides smooth probability frontiers.
2. **K-Nearest Neighbors (KNN):**
   - Non-parametric instance-based algorithm ($k=5$).
   - Computes Euclidean distance in feature space to identify majority vote.
3. **Decision Tree:**
   - Non-linear tree-based classifier recursively partitioning feature space using Gini impurity.
4. **Random Forest:**
   - Ensemble bagging model with 100 randomized decision trees.
   - Reduces variance and prevents overfitting.
5. **Support Vector Machine (SVM):**
   - Radial Basis Function (RBF) kernel projecting features into higher dimensions to construct optimal maximum-margin hyperplanes.

---

## 6. Project Architecture

```text
       User (Web Browser)
               │
               ▼
   [HTML5 / CSS3 / JavaScript]
   (Client Validation & Sample Presets)
               │
          HTTP POST /predict
               ▼
         Flask Backend
   (Input Validation & Sanitization)
               │
               ▼
      Trained ML Model
   (Joblib Serialized Binary)
               │
               ▼
      Prediction & Confidence
               │
               ▼
       Predicted Iris Species
    (Setosa / Versicolor / Virginica)
```

---

## 7. Project Structure

```text
irisflower_classification/
│
├── app.py                      # Flask web application & prediction server
├── train.py                    # Complete ML training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Detailed project documentation
├── .gitignore                  # Git ignore rules
│
├── model/
│   ├── iris_model.pkl          # Serialized top-performing model
│   └── model_info.json         # Model metrics and feature schema
│
├── dataset/
│   └── README.md               # Dataset background & botanical citations
│
├── notebooks/
│   └── iris_analysis.py        # Standalone data science exploration script
│
├── outputs/                    # Auto-generated EDA & comparison charts
│   ├── eda_pairplot.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_sepal_scatter.png
│   ├── eda_petal_scatter.png
│   ├── eda_class_distribution.png
│   ├── confusion_matrices.png
│   └── model_comparison.csv
│
├── tests/
│   ├── test_model.py           # Unit tests for model loading and inference
│   └── test_app.py             # Integration tests for Flask routes
│
├── static/
│   ├── style.css               # Modern responsive design system
│   └── script.js               # Client validation and UI interactivity
│
└── templates/
    ├── index.html              # Main prediction interface
    └── result.html             # Detailed prediction outcome card
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10–3.12 recommended)
- `pip` package manager

### 1. Clone or Open the Project
Navigate into the project directory:
```bash
cd irisflower_classification
```

### 2. Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Models
Execute the training pipeline:
```bash
python train.py
```
This command:
- Explores and validates the Iris dataset
- Generates 5 EDA charts and a confusion matrix in `outputs/`
- Trains all 5 algorithms on an 80/20 stratified split
- Exports `outputs/model_comparison.csv`
- Saves the top-performing model to `model/iris_model.pkl` and metadata to `model/model_info.json`
- Prints the terminal training summary

### 5. Launch the Web Application
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

### 6. Run the Test Suite
```bash
pytest -v
```

---

## 9. Example Usage

### Example Input
| Feature | Measurement |
| :--- | :--- |
| Sepal Length | `5.1 cm` |
| Sepal Width | `3.5 cm` |
| Petal Length | `1.4 cm` |
| Petal Width | `0.2 cm` |

### Example Output
```text
Prediction Result:
🌸 Iris Setosa

Prediction Confidence:
100.00%

Taxonomic Characteristics:
Characterized by distinctly small, narrow petals and wide sepals.
Linearly separable from other Iris species.
```

---

## 10. Model Evaluation Metrics

Each model is evaluated against the held-out test partition ($n=30$, stratified) using:

- **Accuracy:** Proportion of overall correct classifications:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision (Weighted):** Measure of classifier exactness per class weighted by support:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Weighted):** Measure of classifier completeness per class:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score (Weighted):** Harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix:** $3 \times 3$ matrix contrasting ground-truth classes against model predictions to identify misclassifications.

### Benchmark Results
| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors** | **100.00%** | **1.000** | **1.000** | **1.000** |
| **Logistic Regression** | **96.67%** | **0.970** | **0.967** | **0.967** |
| **Support Vector Machine** | **96.67%** | **0.970** | **0.967** | **0.967** |
| **Decision Tree** | **93.33%** | **0.933** | **0.933** | **0.933** |
| **Random Forest** | **90.00%** | **0.902** | **0.900** | **0.900** |

---

## 11. Future Enhancements

- [ ] **Cloud Deployment:** Containerization and deployment to AWS Elastic Beanstalk, Render, or GCP Cloud Run.
- [ ] **Hyperparameter Tuning:** Automated grid search and cross-validation across tree depths and kernel parameters.
- [ ] **Dataset Expansion:** Extend support to custom multi-species botanical databases.
- [ ] **REST API Swagger Docs:** OpenAPI / Swagger documentation for programmatic API consumers.
- [ ] **Containerization:** Ready-to-run `Dockerfile` and `docker-compose.yml`.

---

## 12. License & Academic Attribution
Developed for academic machine learning demonstrations and final year engineering projects. Based on Ronald A. Fisher's botanical Iris taxonomy.

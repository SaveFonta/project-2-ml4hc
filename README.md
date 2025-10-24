# 🧠 Project 2 – Interpretable and Explainable Machine Learning for Healthcare (ML4HC)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project explores interpretable and explainable machine learning methods in healthcare.  
It is structured around two main applications:

1. **Heart failure prediction** from tabular clinical data  
2. **Pneumonia detection** from chest X-ray images  

The goal is to understand how both **model design** and **post-hoc explainability techniques** can provide insights into predictions made by machine learning models — ensuring that models are not only accurate but also transparent and trustworthy for medical applications.

---

## 📦 Project Organization

```

├── LICENSE
├── Makefile
├── README.md
│
├── data
│   ├── external       <- Data from Kaggle or third-party sources
│   ├── interim        <- Intermediate transformed data
│   ├── processed      <- Cleaned and ready-to-use data for modeling
│   └── raw            <- Original, immutable data dumps
│
├── docs               <- Documentation and project handouts
│
├── models             <- Trained models, predictions, and evaluation summaries
│
├── notebooks          <- Jupyter notebooks (numbered and descriptive)
│   ├── part1.ipynb                <- Heart Failure Prediction (Tabular)
│   └── part2/1_fb_part2.ipynb     <- Pneumonia Detection (Images)
│
├── proj2/              <- Main source code
│   ├── **init**.py
│   ├── config.py               <- Configuration variables and constants
│   ├── dataset.py              <- Dataset download and setup logic
│   ├── features.py             <- Feature engineering utilities
│   ├── modeling/
│   │   ├── train.py            <- Model training pipeline
│   │   ├── predict.py          <- Inference and evaluation scripts
│   │   └── **init**.py
│   ├── part2/
│   │   ├── dataloader.py       <- Dataset and DataLoader for X-ray images
│   │   └── training.py         <- CNN training and evaluation loops
│   └── plots.py                <- Visualization and interpretability plots
│
├── reports/
│   ├── figures/                <- Generated plots and saliency maps
│   └── analysis/               <- Reports and summaries
│
├── requirements.txt
├── setup.cfg
└── pyproject.toml

````

---

## ⚙️ Setup

It is **strongly recommended** to install dependencies in a virtual environment to ensure reproducibility.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🔧 Dataset Setup & Configuration

The project requires datasets from **Kaggle** for both parts.
Before downloading, create a `.env` file in the project root (same level as `proj2/`) and set your Kaggle API credentials:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
KAGGLEHUB_CACHE=data/external/
```

You can use the provided `.env.example` as a template.
More information: [Kaggle API documentation](https://www.kaggle.com/docs/api)

### Dataset Download

The dataset setup script automatically:

* Loads credentials from `.env`
* Uses `kagglehub` to download datasets
* Caches files in `KAGGLEHUB_CACHE`
* Checks for local copies before re-downloading

Managed datasets:

* `data/external/datasets/fedesoriano` → *Heart Failure Prediction*
* `data/external/datasets/paultimothymooney` → *Chest X-Ray Pneumonia*

To download the datasets:

```bash
python -m proj2.dataset
```

Or, use the provided VSCode launch configuration for convenience.
This ensures all team members share a consistent, reproducible environment.

---

## 🚀 How to Run the Project

The project is divided into two main parts, each with separate datasets, models, and explainability methods.

---

### 🫀 Part 1 – Heart Failure Prediction (Tabular Data)

This part focuses on **interpretable models** for tabular clinical data using the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

1. **Exploratory Data Analysis (EDA)**

   * Inspect feature distributions, handle missing values and outliers, and identify class imbalance.

2. **Logistic Lasso Regression**

   * Train an L1-regularized logistic model to determine key features.
   * Visualize feature importances and discuss interpretability.
   * Evaluate using **F1-score** and **Balanced Accuracy**.

3. **Multi-Layer Perceptron (MLP) + SHAP**

   * Train a simple MLP classifier.
   * Use **SHAP** values to explain individual predictions and global feature importance.
   * Compare feature attributions with Lasso Regression results.

4. **Neural Additive Models (NAMs)**

   * Implement and train a **NAM** for interpretable nonlinear modeling.
   * Visualize feature-wise contributions and compare against logistic and MLP models.

📓 Notebook:
`notebooks/part1.ipynb`

---

### 🩻 Part 2 – Pneumonia Detection (Chest X-Ray Images)

This part focuses on **explainable deep learning** using CNNs trained on the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

1. **Data Exploration & Preprocessing**

   * Visualize healthy vs. pneumonia samples.
   * Discuss potential biases (e.g., scanner artifacts).
   * Prepare train/test splits and data augmentations.

2. **CNN Classifier**

   * Train a convolutional neural network for binary classification.
   * Evaluate test performance and report accuracy, precision, recall, and F1-score.

3. **Post-hoc Explainability**

   * **Integrated Gradients:**
     Compute pixel attribution maps highlighting influential image regions.
   * **Grad-CAM:**
     Visualize class-discriminative heatmaps and compare consistency with Integrated Gradients.
   * **Data Randomization Test:**
     Assess the reliability of attribution methods by randomizing training labels and comparing maps.

📓 Notebook:
`notebooks/part2/1_fb_part2.ipynb`

---

## 🧩 Recommended VSCode Extension

Install [**Colorful Comments**](https://marketplace.visualstudio.com/items?itemName=ParthR2031.colorful-comments) for an enhanced development experience.
It highlights important comment keywords, helping you quickly navigate TODOs and explanations.

---

## 🧠 Key Concepts Explored

* **Interpretability vs. Explainability** in ML
* **Feature attribution** (Lasso, SHAP, NAMs)
* **Post-hoc explanation methods** (Integrated Gradients, Grad-CAM)
* **Sanity Checks** for interpretability reliability
* **Model transparency** and trust in healthcare AI

---


## 📜 License

Distributed under an open-source license (see `LICENSE`).

---

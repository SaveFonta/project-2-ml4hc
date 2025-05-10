# Project 2 - ML4HC

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         proj2 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── proj2   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes proj2 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    |
    ├── part2
    │   ├── dataloader.py <- Dataset and DataLoader utilities for part 2
    │   └── training.py <- CNN training loop for part 2
    │
    └── plots.py                <- Code to create visualizations
```

---

## 🔧 Dataset Setup & Configuration

To get started, install the required Python packages:

`pip install -r requirements.txt`

Then, create a `.env` file in the project root (at the same level as `proj2/`) and configure your Kaggle credentials along with a custom cache directory for downloads. You can use the provided `.env.example` as a starting point and include your personal details (more info at [this link](https://www.kaggle.com/docs/api))

The dataset download script (`proj2/dataset.py`) is designed to:

- Automatically load credentials from the `.env` file
- Use kagglehub to download datasets
- Store downloaded files in the directory defined by `KAGGLEHUB_CACHE`
- Check if the datasets already exist locally before downloading

By default, it manages the following datasets:

- `data/external/datasets/fedesoriano` – Heart Failure Prediction
- `data/external/datasets/paultimothymooney` – Chest X-Ray Pneumonia

If these folders are not found, the script will automatically download them and extract the datasets into them. This ensures that all team members have a consistent and reproducible environment.
To run this script, ensure your working directory is the project root, then execute:

`python -m proj2.dataset`

Or, use the provided VSCode launch configuration for convenience.

---

## 🚀 How to Run the Project

This project is organized into two main parts. The first part addresses heart failure prediction based on tabular data, while the second part is about pneunomia prediction from X-Ray images. Follow the instructions below to reproduce the results.

---

### ⚙️ Setup

Install dependencies (recommended via virtual environment):

```
pip install -r requirements.txt
```
---

### 🫀 Running Part 1

After downloading the dataset as described above, just open and run the following notebook:
📓 `notebooks/part1.ipynb`

---

### 🩻 Running Part 2

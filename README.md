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

`pip install kagglehub python-dotenv loguru tqdm`

Then, create a `.env` file in the project root (at the same level as `proj2/`) and configure your Kaggle credentials along with a custom cache directory for downloads. You can use the provided `.env.example` as a starting point.

The dataset download script (`proj2/dataset.py`) is designed to:

- Automatically load credentials from the `.env` file
- Use kagglehub to download datasets
- Store downloaded files in the directory defined by `KAGGLEHUB_CACHE`
- Check if the datasets already exist locally before downloading

By default, it manages the following datasets:

- `data/external/datasets/fedesoriano` – Heart Failure Prediction
- `data/external/datasets/paultimothymooney` – Chest X-Ray Pneumonia

If these folders are not found, the script will download and extract the datasets into them. This ensures that all team members have a consistent and reproducible environment.
To run the script, ensure your working directory is the project root, then execute:

`python -m proj2.dataset`

Or, use the provided VSCode launch configuration for convenience.

### 📂 Additional Required Folder: `data/raw`

To support saving preprocessed NumPy datasets (used for faster loading and reproducibility), you must also create the following folder manually:

```
data/raw/
```

This folder is used to store compressed `.npz` versions of the processed image data. These files are automatically created the first time you run the dataset loader (see `get_data_set_loader` in `proj2/dataloader.py`) if they do not already exist. The folder is **not versioned in Git** (it's listed in `.gitignore`) to avoid bloating the repository with large files.

**Make sure to create this folder** before running training or visualization scripts that depend on preprocessed data.

You can create it manually or run the following command from the project root:

```bash
mkdir -p data/raw
```

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
import os
import kagglehub

# Since dataset.py is in proj2/ and .env is one level up:
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Get credentials
username = os.environ.get("KAGGLE_USERNAME")
key = os.environ.get("KAGGLE_KEY")

if not username or not key:
    raise RuntimeError("Kaggle credentials not found.")

print(f"Authenticated as: {username}")

base_dir = Path(__file__).resolve().parent.parent / "data" / "external" / "datasets"
path_1 = base_dir / "fedesoriano"
path_2 = base_dir / "paultimothymooney"

print("Downloading datasets...")

# Download latest version (Part 1)
if not path_1.exists():
    path = kagglehub.dataset_download(
        "fedesoriano/heart-failure-prediction", path="heart.csv"
    )

    print("Path to dataset files:", path)
else:
    print("Dataset 1 already downloaded.")

# Download latest version (Part 2)
if not path_2.exists():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

    print("Path to dataset files:", path)
else:
    print("Dataset 2 already downloaded.")

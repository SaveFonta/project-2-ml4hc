#####################################################################################################
# * This script downloads datasets from Kaggle using the Kaggle API and stores them in a specified directory.
# * It uses the `kagglehub` library to handle the downloading process.
# * The script first loads environment variables from a `.env` file to get the Kaggle credentials.
####################################################################################################

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

# Create folder ./data/raw if it doesn't exist, since it's needed for future steps
raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
if not raw_dir.exists():
    os.makedirs(raw_dir)
    print(f"Created directory: {raw_dir}")

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

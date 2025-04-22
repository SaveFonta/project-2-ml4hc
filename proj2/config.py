###################################################################
# * This file is the file to be imported everytime in the project.
# * It contains the imports, the paths, and the configuration for the correct import of project modules.
# * It also loads the environment variables from the .env file.
##################################################################

from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Load environment variables from .env file if it exists
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "proj2" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

HEART_DATA_FILE = (
    DATA_DIR
    / "external"
    / "datasets"
    / "fedesoriano"
    / "heart-failure-prediction"
    / "versions"
    / "1"
    / "heart.csv"
)


MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

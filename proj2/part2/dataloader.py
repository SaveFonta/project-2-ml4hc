#####################################################################################################
# * This script loads the data for the second part of the project into Datasets and DataLoaders.
# * It uses the `torchvision` library to handle the dataset loading process.
####################################################################################################

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

RAW_DIR = Path.cwd().parent.parent / "data" / "raw"
BASE_DIR = (
    Path.cwd().parent.parent
    / "data"
    / "external"
    / "datasets"
    / "paultimothymooney"
    / "chest-xray-pneumonia"
    / "versions"
    / "2"
    / "chest_xray"
)
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"
print(f"BASE_DIR: {BASE_DIR}")


def get_data_set_loader(image_size=(224, 224), batch_size=32):

    def prep_and_save(folder: Path, out_file: Path):
        # Load from ImageFolder, apply transform, collect arrays
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        ds = datasets.ImageFolder(root=folder, transform=transform)
        imgs, labels = [], []
        for img, lbl in ds:
            imgs.append(img.numpy())
            labels.append(lbl)
        np.savez_compressed(
            out_file,
            images=np.stack(imgs),
            labels=np.array(labels),
        )
        return out_file

    # Check existence of raw files
    raw_train = RAW_DIR / "train.npz"
    raw_val = RAW_DIR / "val.npz"
    raw_test = RAW_DIR / "test.npz"
    if not (raw_train.exists() and raw_val.exists() and raw_test.exists()):
        print("🌱 Preprocessing raw images and saving to data/raw...")
        prep_and_save(BASE_DIR / "train", raw_train)
        prep_and_save(BASE_DIR / "val", raw_val)
        prep_and_save(BASE_DIR / "test", raw_test)

    # Load from npz files
    def load_npz(in_file: Path):
        npz = np.load(in_file)
        images = torch.from_numpy(npz["images"])
        labels = torch.from_numpy(npz["labels"])
        return TensorDataset(images, labels)

    train_ds = load_npz(raw_train)
    val_ds = load_npz(raw_val)
    test_ds = load_npz(raw_test)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("Classes:", datasets.ImageFolder(BASE_DIR / "train").classes)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val   samples: {len(val_ds)}")
    print(f"Test  samples: {len(test_ds)}")

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

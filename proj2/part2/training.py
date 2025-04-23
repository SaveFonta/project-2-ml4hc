from proj2.config import PROJ_ROOT
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np


def train_CNN(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion=torch.nn.BCEWithLogitsLoss(),
    num_epochs=10,
    device=None,
    run_name="",
):

    # Prepare the directory
    checkpoint_dir = PROJ_ROOT / "models" / "cnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    # Outer epoch loop
    # Top‐level epoch bar (position 0)
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_bar:
        # ----- Training -----
        model.train()
        running_loss = 0.0

        # Sample‑level progress bar
        train_pbar = tqdm(
            total=len(train_loader.dataset),
            desc=f" Epoch {epoch}/{num_epochs} ▶ TRAIN ",
            unit="img",
            position=1,
            leave=True,
        )
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            # advance by number of samples in this batch
            train_pbar.update(batch_size)
            train_pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_pbar.close()

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        val_pbar = tqdm(
            total=len(val_loader.dataset),
            desc=f" Epoch {epoch}/{num_epochs} ◀ VALID ",
            unit="img",
            position=2,
            leave=True,
        )
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_targets.extend(labels.cpu().numpy().flatten())

                val_pbar.update(images.size(0))
                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_pbar.close()

        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        binary_preds = (all_preds >= 0.5).astype(int)
        cm = confusion_matrix(all_targets, binary_preds)

        epoch_metrics = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "AUROC": roc_auc_score(all_targets, all_preds),
            "AUPRC": average_precision_score(all_targets, all_preds),
            "Accuracy": accuracy_score(all_targets, binary_preds),
            "Precision": precision_score(all_targets, binary_preds),
            "Recall": recall_score(all_targets, binary_preds),
            "F1": f1_score(all_targets, binary_preds),
            "Confusion Matrix": cm,
        }

        # Update epoch bar with summary
        epoch_bar.set_postfix(
            {
                "tr_loss": f"{epoch_metrics['train_loss']:.4f}",
                "vl_loss": f"{epoch_metrics['val_loss']:.4f}",
                "AUROC": f"{epoch_metrics['AUROC']:.3f}",
                "Acc": f"{epoch_metrics['Accuracy']:.3f}",
                "F1": f"{epoch_metrics['F1']:.3f}",
                "Prec": f"{epoch_metrics['Precision']:.3f}",
                "Rec": f"{epoch_metrics['Recall']:.3f}",
                "TP": f"{cm[1, 1]}",
                "TN": f"{cm[0, 0]}",
                "FP": f"{cm[0, 1]}",
                "FN": f"{cm[1, 0]}",
            }
        )

        # ----- Checkpoint every 5 epochs -----
        if epoch % 5 == 0:
            ckpt_path = checkpoint_dir / f"{run_name}_checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"\n🔖 Checkpoint saved: {ckpt_path}")

    # ----- Save final model -----
    final_path = checkpoint_dir / f"{run_name}_model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Final model saved: {final_path}")

    print("\nTraining complete.")


def eval_CNN(model, test_loader, criterion=torch.nn.BCEWithLogitsLoss(), device=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            # Move inputs & targets to device, reshape targets to [B,1]
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            # Forward pass (raw logits)
            logits = model(images)

            # Compute loss on raw logits
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

            # Convert logits -> probabilities for metrics
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    binary_preds = (all_preds >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "AUROC": roc_auc_score(all_targets, all_preds),
        "AUPRC": average_precision_score(all_targets, all_preds),
        "Accuracy": accuracy_score(all_targets, binary_preds),
        "Precision": precision_score(all_targets, binary_preds),
        "Recall": recall_score(all_targets, binary_preds),
        "F1 Score": f1_score(all_targets, binary_preds),
        "Confusion Matrix": confusion_matrix(all_targets, binary_preds),
    }

    # Include average test loss if requested
    if criterion is not None:
        metrics["Test Loss"] = total_loss / len(test_loader.dataset)

    # Print results
    print("\n✅ Test Set Performance:")
    if "Test Loss" in metrics:
        print(f"  Test Loss       = {metrics['Test Loss']:.4f}")
    print(f"  AUROC           = {metrics['AUROC']:.4f}")
    print(f"  AUPRC           = {metrics['AUPRC']:.4f}")
    print(f"  Accuracy        = {metrics['Accuracy']:.4f}")
    print(f"  Precision       = {metrics['Precision']:.4f}")
    print(f"  Recall          = {metrics['Recall']:.4f}")
    print(f"  F1 Score        = {metrics['F1 Score']:.4f}")
    print("  Confusion Matrix:")
    print(metrics["Confusion Matrix"])

    return metrics

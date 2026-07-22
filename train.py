import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMWinPredictor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smooth_labels(y, smoothing=0.05):
    """Applies label smoothing to binary targets."""
    return y * (1.0 - smoothing) + 0.5 * smoothing


def main():
    processed_path = "artifacts/processed_data.npz"
    mappings_path = "artifacts/mappings.json"

    if not os.path.exists(processed_path) or not os.path.exists(mappings_path):
        raise FileNotFoundError("Processed dataset or metadata mappings missing. Run 'python data_preprocessing.py' first.")

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    data = np.load(processed_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    device = get_device()
    print(f"Using compute device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    num_players = mappings.get("num_players", 1000)
    num_teams = mappings.get("num_teams", 30)
    num_venues = mappings.get("num_venues", 100)

    model = LSTMWinPredictor(
        num_players=num_players,
        num_teams=num_teams,
        num_venues=num_venues,
        hidden_size=48,
        dropout=0.5
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    epochs = 20
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Starting regularized model training for {epochs} epochs on {len(train_dataset)} sequence samples...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_smoothed = smooth_labels(yb, smoothing=0.05)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb_smoothed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            predicted = (preds >= 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss += loss.item() * xb.size(0)
                predicted = (preds >= 0.5).float()
                val_correct += (predicted == yb).sum().item()
                val_total += yb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] - "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("artifacts", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/model.pt")

    with open("artifacts/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Model successfully trained! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMWinPredictor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    processed_path = "artifacts/processed_data.npz"
    model_path = "artifacts/model.pt"
    mappings_path = "artifacts/mappings.json"

    if not os.path.exists(processed_path) or not os.path.exists(model_path) or not os.path.exists(mappings_path):
        raise FileNotFoundError("Required artifacts missing. Please run preprocessing and training first.")

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    data = np.load(processed_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    device = get_device()
    print(f"Evaluating model on device: {device}")

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_players = mappings.get("num_players", 1000)
    num_teams = mappings.get("num_teams", 30)
    num_venues = mappings.get("num_venues", 100)

    model = LSTMWinPredictor(
        num_players=num_players,
        num_teams=num_teams,
        num_venues=num_venues,
        hidden_size=48
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.BCELoss()
    test_loss = 0.0
    y_test_prob_list = []
    y_test_true_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = model(xb)
            loss = criterion(probs, yb)
            test_loss += loss.item() * xb.size(0)

            y_test_prob_list.extend(probs.squeeze(1).cpu().numpy().tolist())
            y_test_true_list.extend(yb.squeeze(1).cpu().numpy().tolist())

    test_loss /= len(test_dataset)
    y_test_prob = np.array(y_test_prob_list)
    y_test_true = np.array(y_test_true_list)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    test_accuracy = float(accuracy_score(y_test_true, y_test_pred))
    test_roc_auc = float(roc_auc_score(y_test_true, y_test_prob))
    test_precision = float(precision_score(y_test_true, y_test_pred, zero_division=0))
    test_recall = float(recall_score(y_test_true, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test_true, y_test_pred, zero_division=0))

    metrics = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1_score": round(test_f1, 4),
    }

    with open("artifacts/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ HIGH ACCURACY MODEL EVALUATION REPORT ================")
    print(f" Test Loss:        {metrics['test_loss']:.4f}")
    print(f" Test Accuracy:    {metrics['test_accuracy'] * 100:.2f}%")
    print(f" Test ROC-AUC:     {metrics['test_roc_auc']:.4f}")
    print(f" Test Precision:   {metrics['test_precision']:.4f}")
    print(f" Test Recall:      {metrics['test_recall']:.4f}")
    print(f" Test F1 Score:    {metrics['test_f1_score']:.4f}")
    print("========================================================================\n")
    print("Sample predicted probabilities (first 10):")
    for i, p in enumerate(y_test_prob[:10]):
        print(f" Sample {i+1:02d}: {p:.4f} (Actual: {int(y_test_true[i])})")


if __name__ == "__main__":
    main()
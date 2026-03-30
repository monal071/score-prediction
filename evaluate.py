import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

from model import LSTMWinPredictor


def main():
    data = np.load('artifacts/processed_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    input_size = X_test.shape[2]
    hidden_size = 64

    model = LSTMWinPredictor(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load('artifacts/model.pt', map_location='cpu'))
    model.eval()

    y_test_prob_list = []
    y_test_true_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            probs = model(xb)
            y_test_prob_list.extend(probs.squeeze(1).cpu().numpy().tolist())
            y_test_true_list.extend(yb.squeeze(1).cpu().numpy().tolist())

    y_test_prob = np.array(y_test_prob_list)
    y_test_true = np.array(y_test_true_list)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_roc_auc = roc_auc_score(y_test_true, y_test_prob)

    print('Test Accuracy:', test_accuracy)
    print('Test ROC-AUC:', test_roc_auc)
    print('Predicted probabilities sample:', y_test_prob[:10])


if __name__ == '__main__':
    main()
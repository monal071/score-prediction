import numpy as np
import torch
from model import LSTMWinPredictor


def main():
    data = np.load('artifacts/processed_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    if len(X_test) == 0:
        raise ValueError('X_test is empty. Run data_preprocessing.py first.')

    input_size = X_test.shape[2]
    hidden_size = 64

    model = LSTMWinPredictor(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load('artifacts/model.pt', map_location='cpu'))
    model.eval()

    sample_idx = 0
    x = torch.tensor(X_test[sample_idx], dtype=torch.float32).unsqueeze(0)
    y_true = int(y_test[sample_idx])

    with torch.no_grad():
        prob = float(model(x).item())

    pred = int(prob >= 0.5)

    print('Sample index:', sample_idx)
    print('True label:', y_true)
    print('Predicted win probability:', prob)
    print('Predicted class (1=win, 0=loss):', pred)


if __name__ == '__main__':
    main()
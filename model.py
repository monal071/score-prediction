import torch
import torch.nn as nn


class LSTMWinPredictor(nn.Module):
    """
    LSTM sequence classifier for second-innings cricket win probability prediction.
    
    Args:
        input_size (int): Number of input features per ball sequence.
        hidden_size (int): Number of hidden units in the LSTM layer.
        dropout (float): Optional dropout rate between LSTM and FC layer.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take the output of the last time step in the sequence
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

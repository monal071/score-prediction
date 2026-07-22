import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Additive attention mechanism over LSTM sequence time steps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        scores = self.attn(lstm_output)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        return context


class LSTMWinPredictor(nn.Module):
    """
    Regularized Attention-BiLSTM Win Predictor with Entity Embeddings.
    """
    def __init__(
        self,
        num_players: int = 1000,
        num_teams: int = 30,
        num_venues: int = 100,
        hidden_size: int = 48,
        dropout: float = 0.5
    ):
        super().__init__()

        # Compact Entity Embeddings with higher regularization
        self.player_embed = nn.Embedding(num_embeddings=num_players + 2, embedding_dim=8, padding_idx=0)
        self.team_embed = nn.Embedding(num_embeddings=num_teams + 2, embedding_dim=6, padding_idx=0)
        self.venue_embed = nn.Embedding(num_embeddings=num_venues + 2, embedding_dim=6, padding_idx=0)

        self.emb_dropout = nn.Dropout(dropout)

        # Input dimension: 6 (batting_team) + 6 (bowling_team) + 6 (venue) + 8 (batter) + 8 (bowler) + 9 continuous = 44
        total_input_size = 6 + 6 + 6 + 8 + 8 + 9

        self.lstm = nn.LSTM(
            input_size=total_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        bilstm_hidden_dim = hidden_size * 2
        self.attention = AdditiveAttention(hidden_dim=bilstm_hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(bilstm_hidden_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batting_team_idx = x[:, :, 0].long().clamp(min=0)
        bowling_team_idx = x[:, :, 1].long().clamp(min=0)
        venue_idx        = x[:, :, 2].long().clamp(min=0)
        batter_idx       = x[:, :, 3].long().clamp(min=0)
        bowler_idx       = x[:, :, 4].long().clamp(min=0)

        cont_features = x[:, :, 5:]

        # Embeddings + Embedding Dropout
        batting_team_emb = self.emb_dropout(self.team_embed(batting_team_idx))
        bowling_team_emb = self.emb_dropout(self.team_embed(bowling_team_idx))
        venue_emb        = self.emb_dropout(self.venue_embed(venue_idx))
        batter_emb       = self.emb_dropout(self.player_embed(batter_idx))
        bowler_emb       = self.emb_dropout(self.bowler_embed_or_player(bowler_idx))

        combined = torch.cat([
            batting_team_emb,
            bowling_team_emb,
            venue_emb,
            batter_emb,
            bowler_emb,
            cont_features
        ], dim=-1)

        lstm_out, _ = self.lstm(combined)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out

    def bowler_embed_or_player(self, bowler_idx):
        return self.player_embed(bowler_idx)

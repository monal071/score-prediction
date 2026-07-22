import argparse
import json
import os
import numpy as np
import torch
from model import LSTMWinPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict win probability for a sample from the test set.")
    parser.add_argument("--sample", type=int, default=0, help="Index of the test sample to evaluate (default: 0)")
    args = parser.parse_args()

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

    if len(X_test) == 0:
        raise ValueError("X_test is empty. Please check data preprocessing.")

    sample_idx = args.sample % len(X_test)

    num_players = mappings.get("num_players", 1000)
    num_teams = mappings.get("num_teams", 30)
    num_venues = mappings.get("num_venues", 100)

    model = LSTMWinPredictor(
        num_players=num_players,
        num_teams=num_teams,
        num_venues=num_venues,
        hidden_size=48
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.tensor(X_test[sample_idx], dtype=torch.float32).unsqueeze(0)
    y_true = int(y_test[sample_idx])

    with torch.no_grad():
        prob = float(model(x).item())

    pred_class = int(prob >= 0.5)
    win_pct = prob * 100.0

    batting_team = "Batting Team"
    bowling_team = "Bowling Team"

    id_to_team = {v: k for k, v in mappings.get("batting_team_to_id", {}).items()}
    last_ball = X_test[sample_idx][-1]
    bat_id = int(last_ball[0])
    bowl_id = int(last_ball[1])
    batting_team = id_to_team.get(bat_id, "Batting Team")
    bowling_team = id_to_team.get(bowl_id, "Bowling Team")

    bar_len = 20
    filled = int(round(prob * bar_len))
    bar = "=" * filled + "-" * (bar_len - filled)

    print("\n================ MATCH WIN PREDICTION ================")
    print(f" Sample Index:          {sample_idx} / {len(X_test) - 1}")
    print(f" Batting Team:          {batting_team}")
    print(f" Bowling Team:          {bowling_team}")
    print(f" Ground Truth (Actual): {'WIN' if y_true == 1 else 'LOSS'} ({y_true})")
    print(f" Model Predicted Win Prob: {win_pct:.2f}%")
    print(f" Confidence Bar:        [{bar}]")
    print(f" Decision:              {'WIN' if pred_class == 1 else 'LOSS'}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
import argparse
import json
import os
import random
import numpy as np
import pandas as pd


def trim_and_engineer_features(df):
    """Filters data for 2nd innings and engineers rate and momentum features."""
    df_2nd = df[df["innings"] == 2].copy()
    
    # Raw match metrics
    runs_left_raw = (df_2nd["runs_target"] - df_2nd["team_runs"]).clip(lower=0)
    balls_left_raw = (120 - df_2nd["team_balls"]).clip(lower=1)
    team_balls_raw = df_2nd["team_balls"].clip(lower=1)

    # Rates & pressure features
    rrr_raw = (runs_left_raw * 6.0) / balls_left_raw
    crr_raw = (df_2nd["team_runs"] * 6.0) / team_balls_raw
    rrr_crr_diff_raw = rrr_raw - crr_raw

    df_2nd["runs_left_raw"] = runs_left_raw
    df_2nd["rrr_raw"] = rrr_raw
    df_2nd["crr_raw"] = crr_raw
    df_2nd["rrr_crr_diff_raw"] = rrr_crr_diff_raw
    df_2nd["wickets_left_raw"] = 10 - df_2nd["team_wicket"]

    # Rolling momentum over last 12 balls per match
    df_2nd["runs_last_12"] = df_2nd.groupby("match_id")["team_runs"].diff(periods=12).fillna(df_2nd["team_runs"]).clip(lower=0)
    df_2nd["wickets_last_12"] = df_2nd.groupby("match_id")["team_wicket"].diff(periods=12).fillna(df_2nd["team_wicket"]).clip(lower=0)

    # Match outcome target (1 = win, 0 = loss)
    df_2nd["results"] = (df_2nd["match_won_by"] == df_2nd["batting_team"]).astype(int)

    columns_to_keep = [
        "match_id", "batting_team", "bowling_team", "venue", "batter", "bowler",
        "runs_left_raw", "team_balls", "team_wicket", "runs_target",
        "rrr_raw", "crr_raw", "rrr_crr_diff_raw", "wickets_left_raw",
        "runs_last_12", "wickets_last_12", "results"
    ]

    missing_cols = [col for col in columns_to_keep if col not in df_2nd.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataset: {missing_cols}")

    return df_2nd.loc[:, columns_to_keep].copy()


def normalize_and_encode(df2, df_raw):
    """Encodes categorical features, normalizes continuous features, and exports metadata."""
    players = pd.unique(pd.concat([df_raw["batter"], df_raw["bowler"]], ignore_index=True).dropna())
    player_to_id = {name: idx + 1 for idx, name in enumerate(players)}

    batting_team_values = pd.unique(df_raw["batting_team"].dropna())
    bowling_team_values = pd.unique(df_raw["bowling_team"].dropna())
    venue_values = pd.unique(df_raw["venue"].dropna())

    batting_team_to_id = {name: idx + 1 for idx, name in enumerate(batting_team_values)}
    bowling_team_to_id = {name: idx + 1 for idx, name in enumerate(bowling_team_values)}
    venue_to_id = {name: idx + 1 for idx, name in enumerate(venue_values)}

    # Categorical encodings (1-indexed integers)
    df2["batter"] = df2["batter"].map(player_to_id).fillna(0).astype(int)
    df2["bowler"] = df2["bowler"].map(player_to_id).fillna(0).astype(int)
    df2["batting_team"] = df2["batting_team"].map(batting_team_to_id).fillna(0).astype(int)
    df2["bowling_team"] = df2["bowling_team"].map(bowling_team_to_id).fillna(0).astype(int)
    df2["venue"] = df2["venue"].map(venue_to_id).fillna(0).astype(int)

    max_runs_target = float(df_raw["runs_target"].max()) if not df_raw.empty else 250.0

    # Normalized continuous features
    df2["runs_left"] = df2["runs_left_raw"] / max_runs_target
    df2["team_balls"] = df2["team_balls"] / 120.0
    df2["team_wicket"] = df2["team_wicket"] / 10.0
    df2["rrr"] = (df2["rrr_raw"] / 24.0).clip(lower=0, upper=2.0)
    df2["crr"] = (df2["crr_raw"] / 24.0).clip(lower=0, upper=2.0)
    df2["rrr_crr_diff"] = (df2["rrr_crr_diff_raw"] / 24.0).clip(lower=-2.0, upper=2.0)
    df2["wickets_left"] = df2["wickets_left_raw"] / 10.0
    df2["runs_last_12"] = (df2["runs_last_12"] / 36.0).clip(lower=0, upper=1.0)
    df2["wickets_last_12"] = (df2["wickets_last_12"] / 5.0).clip(lower=0, upper=1.0)

    os.makedirs("artifacts", exist_ok=True)

    mappings = {
        "player_to_id": player_to_id,
        "batting_team_to_id": batting_team_to_id,
        "bowling_team_to_id": bowling_team_to_id,
        "venue_to_id": venue_to_id,
        "num_players": len(player_to_id),
        "num_teams": len(batting_team_to_id),
        "num_venues": len(venue_to_id),
        "max_runs_target": max_runs_target,
        "cat_cols": ["batting_team", "bowling_team", "venue", "batter", "bowler"],
        "num_cols": [
            "runs_left", "team_balls", "team_wicket",
            "rrr", "crr", "rrr_crr_diff", "wickets_left",
            "runs_last_12", "wickets_last_12"
        ]
    }

    with open("artifacts/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    return df2


def train_test_split_sequences(df2, sequence_length=20, random_seed=42, train_size=0.8):
    """Splits match data into sequence windows for train and test sets."""
    list_of_matches = df2.groupby("match_id")
    match_ids = list(list_of_matches.groups.keys())
    
    random.seed(random_seed)
    random.shuffle(match_ids)

    X_train, X_test = [], []
    y_train, y_test = [], []

    split_index = int(len(match_ids) * train_size)

    # 5 categorical + 9 continuous = 14 total features
    feature_cols = [
        "batting_team", "bowling_team", "venue", "batter", "bowler",
        "runs_left", "team_balls", "team_wicket",
        "rrr", "crr", "rrr_crr_diff", "wickets_left",
        "runs_last_12", "wickets_last_12"
    ]

    for idx, match_id in enumerate(match_ids):
        match_data = list_of_matches.get_group(match_id).reset_index(drop=True)
        features = match_data[feature_cols].values
        label = match_data["results"].iloc[0]

        target_X = X_train if idx < split_index else X_test
        target_y = y_train if idx < split_index else y_test

        num_balls = len(features)

        if num_balls < sequence_length:
            padding = np.zeros((sequence_length - num_balls, features.shape[1]))
            padded_features = np.vstack([features, padding])
            target_X.append(padded_features)
            target_y.append(label)
        else:
            for start_idx in range(num_balls - sequence_length + 1):
                seq = features[start_idx:start_idx + sequence_length]
                target_X.append(seq)
                target_y.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Preprocess IPL ball-by-ball dataset into LSTM sequence windows.")
    parser.add_argument("--input", type=str, default="IPL.csv", help="Path to input CSV dataset file (default: IPL.csv)")
    args = parser.parse_args()

    dataset_path = args.input

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")

    df_raw = pd.read_csv(dataset_path, low_memory=False)
    df_engineered = trim_and_engineer_features(df_raw)
    df_processed = normalize_and_encode(df_engineered, df_raw)

    sequence_length = 20
    random_seed = 42
    train_size = 0.8

    X_train, X_test, y_train, y_test = train_test_split_sequences(
        df_processed, sequence_length=sequence_length, random_seed=random_seed, train_size=train_size
    )

    os.makedirs("artifacts", exist_ok=True)
    np.savez(
        "artifacts/processed_data.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print("Data Preprocessing Completed Successfully!")
    print(f" - Engineered 14 total features per ball (5 categorical + 9 rate/momentum continuous)")
    print(f" - X_train shape: {X_train.shape}")
    print(f" - X_test shape:  {X_test.shape}")
    print(f" - y_train shape: {y_train.shape}")
    print(f" - y_test shape:  {y_test.shape}")


if __name__ == "__main__":
    main()

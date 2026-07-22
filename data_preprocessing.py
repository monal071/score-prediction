import argparse
import json
import os
import random
import numpy as np
import pandas as pd


def trim_data(df):
    """Filters data for 2nd innings and computes features/target columns."""
    df_2nd = df[df["innings"] == 2].copy()
    df_2nd["runs_left"] = df_2nd["runs_target"] - df_2nd["team_runs"]
    df_2nd["results"] = (df_2nd["match_won_by"] == df_2nd["batting_team"]).astype(int)

    columns_to_keep = [
        "match_id", "batting_team", "bowling_team", "venue", "batter", "bowler",
        "runs_left", "team_balls", "team_wicket", "runs_target", "results"
    ]

    missing_cols = [col for col in columns_to_keep if col not in df_2nd.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in dataset: {missing_cols}")

    return df_2nd.loc[:, columns_to_keep].copy()


def normalize_and_encode(df2, df_raw):
    """Encodes categorical features, normalizes numeric features, and saves mappings."""
    players = pd.unique(pd.concat([df_raw["batter"], df_raw["bowler"]], ignore_index=True).dropna())
    player_to_id = {name: idx + 1 for idx, name in enumerate(players)}

    batting_team_values = pd.unique(df_raw["batting_team"].dropna())
    bowling_team_values = pd.unique(df_raw["bowling_team"].dropna())
    venue_values = pd.unique(df_raw["venue"].dropna())

    batting_team_to_id = {name: idx + 1 for idx, name in enumerate(batting_team_values)}
    bowling_team_to_id = {name: idx + 1 for idx, name in enumerate(bowling_team_values)}
    venue_to_id = {name: idx + 1 for idx, name in enumerate(venue_values)}

    df2["batter"] = df2["batter"].map(player_to_id).fillna(0)
    df2["bowler"] = df2["bowler"].map(player_to_id).fillna(0)
    df2["batting_team"] = df2["batting_team"].map(batting_team_to_id).fillna(0)
    df2["bowling_team"] = df2["bowling_team"].map(bowling_team_to_id).fillna(0)
    df2["venue"] = df2["venue"].map(venue_to_id).fillna(0)

    max_runs_target = float(df_raw["runs_target"].max()) if not df_raw.empty else 200.0

    df2["runs_left"] = df2["runs_left"] / max_runs_target
    df2["team_wicket"] = df2["team_wicket"] / 10.0
    df2["team_balls"] = df2["team_balls"] / 120.0

    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Save categorical mappings for downstream inference & Streamlit app
    mappings = {
        "player_to_id": player_to_id,
        "batting_team_to_id": batting_team_to_id,
        "bowling_team_to_id": bowling_team_to_id,
        "venue_to_id": venue_to_id,
        "max_runs_target": max_runs_target,
        "feature_cols": [
            "batting_team", "bowling_team", "venue", "batter", "bowler",
            "runs_left", "team_balls", "team_wicket"
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

    feature_cols = [
        "batting_team", "bowling_team", "venue", "batter", "bowler",
        "runs_left", "team_balls", "team_wicket"
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


def download_from_kaggle(dataset_handle: str) -> str:
    """Downloads dataset from Kaggle using kagglehub if available."""
    try:
        import kagglehub
        print(f"Downloading Kaggle dataset: '{dataset_handle}' via kagglehub...")
        path = kagglehub.dataset_download(dataset_handle)
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in downloaded Kaggle dataset at {path}")
        print(f"Downloaded CSV: {csv_files[0]}")
        return csv_files[0]
    except ImportError:
        raise ImportError(
            "Package 'kagglehub' is required to download Kaggle datasets automatically.\n"
            "Please run: pip install kagglehub"
        )


def main():
    parser = argparse.ArgumentParser(description="Preprocess IPL ball-by-ball dataset into LSTM sequence windows.")
    parser.add_argument("--input", type=str, default="IPL.csv", help="Path or URL to input CSV dataset file (default: IPL.csv)")
    parser.add_argument("--kaggle", type=str, default=None, help="Kaggle dataset handle (e.g. chaitu20/ipl-dataset2008-2025)")
    args = parser.parse_args()

    dataset_path = args.input

    if args.kaggle:
        dataset_path = download_from_kaggle(args.kaggle)
    elif not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_path}' not found.\n"
            "Please place 'IPL.csv' in the project root or specify a dataset via:\n"
            "  python data_preprocessing.py --input path/to/dataset.csv\n"
            "  python data_preprocessing.py --kaggle chaitu20/ipl-dataset2008-2025"
        )

    df_raw = pd.read_csv(dataset_path, low_memory=False)
    df_trimmed = trim_data(df_raw)
    df_processed = normalize_and_encode(df_trimmed, df_raw)

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
    print(f" - Dataset Source: {dataset_path}")
    print(f" - X_train shape:   {X_train.shape}")
    print(f" - X_test shape:    {X_test.shape}")
    print(f" - y_train shape:   {y_train.shape}")
    print(f" - y_test shape:    {y_test.shape}")


if __name__ == "__main__":
    main()

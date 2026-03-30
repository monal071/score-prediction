import pandas as pd
import numpy as np
import random
import os

def train_test(df2,sl,rs,ts):

    list_of_matches = df2.groupby('match_id')
    match_ids = list(list_of_matches.groups.keys())
    random.seed(rs)
    random.shuffle(match_ids)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    seq_length = sl
    split_index = int(len(match_ids) * ts)

    feature_cols = [
        'batting_team', 'bowling_team', 'venue', 'batter', 'bowler',
        'runs_left', 'team_balls', 'team_wicket'
    ]

    for idx, match_id in enumerate(match_ids):
        match_data = list_of_matches.get_group(match_id).reset_index(drop=True)
        features = match_data[feature_cols].values
        label = match_data['results'].iloc[0]

        target_X = X_train if idx < split_index else X_test
        target_y = y_train if idx < split_index else y_test

        num_balls = len(features)

        if num_balls < seq_length:
            padding = np.zeros((seq_length - num_balls, features.shape[1]))
            padded_features = np.vstack([features, padding])
            target_X.append(padded_features)
            target_y.append(label)
        else:
            for start_idx in range(num_balls - seq_length + 1):
                seq = features[start_idx:start_idx + seq_length]
                target_X.append(seq)
                target_y.append(label)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train,X_test,y_train,y_test

def normal(df2,df):
    
    players = pd.unique(pd.concat([df['batter'], df['bowler']], ignore_index=True).dropna())
    player_to_id = {name: idx + 1 for idx, name in enumerate(players)}


    batting_team_values = pd.unique(df['batting_team'].dropna())
    bowling_team_values = pd.unique(df['bowling_team'].dropna())
    venue_values = pd.unique(df['venue'].dropna())

    batting_team_to_id = {name: idx + 1 for idx, name in enumerate(batting_team_values)}
    bowling_team_to_id = {name: idx + 1 for idx, name in enumerate(bowling_team_values)}
    venue_to_id = {name: idx + 1 for idx, name in enumerate(venue_values)}



    df2['batter'] = df2['batter'].map(player_to_id)
    df2['bowler'] = df2['bowler'].map(player_to_id)
    df2['batting_team'] = df2['batting_team'].map(batting_team_to_id)
    df2['bowling_team'] = df2['bowling_team'].map(bowling_team_to_id)
    df2['venue'] = df2['venue'].map(venue_to_id)

    max_runs_target = df['runs_target'].max()

    df2['runs_left'] = df2['runs_left'] / max_runs_target
    df2['team_wicket'] = df2['team_wicket'] / 10.0
    df2['team_balls'] = df2['team_balls'] / 120.0

    return df2

def trim_data(df):
    df=df[df['innings']==2]
    df['runs_left']=df['runs_target']-df['team_runs']
    df['results'] = (df['match_won_by'] == df['batting_team']).astype(int)

    columns_to_keep = [
        'match_id','batting_team', 'bowling_team', 'venue', 'batter', 'bowler', 'runs_left',
        'team_balls', 'team_wicket', 'runs_target', 'results'
    ]

    missing_cols = [col for col in columns_to_keep if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in df: {missing_cols}")

    return df.loc[:, columns_to_keep].copy()

def main():
    df = pd.read_csv('IPL.csv')

    df2 = normal(trim_data(df), df)

    train_size = 0.8
    sequence_len = 20
    random_seed = 42

    X_train, X_test, y_train, y_test = train_test(df2, sequence_len, random_seed, train_size)

    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    np.savez(
        'artifacts/processed_data.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print('X_train:', X_train.shape)
    print('X_test:', X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)


if __name__ == '__main__':
    main()

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from model import LSTMWinPredictor

# Page configuration
st.set_page_config(
    page_title="Cricket Win Predictor - IPL LSTM Model",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high visual aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2e364f;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #171b26 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .win-badge {
        background-color: #10B981;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .loss-badge {
        background-color: #EF4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_mappings():
    model_path = "artifacts/model.pt"
    mappings_path = "artifacts/mappings.json"

    if not os.path.exists(model_path) or not os.path.exists(mappings_path):
        return None, None

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    input_size = len(mappings["feature_cols"])
    hidden_size = 64

    model = LSTMWinPredictor(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, mappings


@st.cache_data
def load_processed_data():
    processed_path = "artifacts/processed_data.npz"
    if not os.path.exists(processed_path):
        return None
    return np.load(processed_path)


def main():
    st.title("🏏 Cricket Win Probability Estimator")
    st.caption("Deep Learning LSTM Sequence Model for IPL 2nd Innings Win Prediction")

    model, mappings = load_model_and_mappings()
    data = load_processed_data()

    if model is None or mappings is None:
        st.warning("⚠️ Model weights or metadata not found in `artifacts/`. Please run the pipeline scripts first:")
        st.code("python data_preprocessing.py\npython train.py\npython evaluate.py", language="bash")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["🎯 Match Scenario Simulator", "📊 Test Dataset Trajectory", "📈 Model Performance & Info"]
    )

    id_to_batting_team = {v: k for k, v in mappings["batting_team_to_id"].items()}
    id_to_bowling_team = {v: k for k, v in mappings["bowling_team_to_id"].items()}
    id_to_venue = {v: k for k, v in mappings["venue_to_id"].items()}
    id_to_player = {v: k for k, v in mappings["player_to_id"].items()}

    # TAB 1: MATCH SCENARIO SIMULATOR
    if app_mode == "🎯 Match Scenario Simulator":
        st.subheader("Simulate 2nd Innings Match Scenario")
        st.write("Configure match parameters to calculate real-time win probability.")

        col1, col2, col3 = st.columns(3)

        teams = list(mappings["batting_team_to_id"].keys())
        venues = list(mappings["venue_to_id"].keys())
        players = list(mappings["player_to_id"].keys())

        with col1:
            batting_team = st.selectbox("Batting Team (Chasing)", teams, index=0)
            bowling_team = st.selectbox("Bowling Team (Defending)", teams, index=min(1, len(teams)-1))
            venue = st.selectbox("Venue", venues, index=0)

        with col2:
            batter = st.selectbox("Current Batter", players, index=0)
            bowler = st.selectbox("Current Bowler", players, index=min(1, len(players)-1))
            target_runs = st.number_input("Target Runs (1st Innings Score + 1)", min_value=50, max_value=300, value=175)

        with col3:
            current_runs = st.number_input("Current Runs Scored", min_value=0, max_value=target_runs, value=110)
            team_balls = st.slider("Balls Bowled (Out of 120)", min_value=1, max_value=120, value=80)
            team_wicket = st.slider("Wickets Lost", min_value=0, max_value=9, value=3)

        # Build 20-ball input sequence (assuming past 20 balls had uniform steady progression)
        max_runs_target = mappings["max_runs_target"]
        runs_left = max(0, target_runs - current_runs)

        bat_id = mappings["batting_team_to_id"].get(batting_team, 0)
        bowl_id = mappings["bowling_team_to_id"].get(bowling_team, 0)
        ven_id = mappings["venue_to_id"].get(venue, 0)
        btr_id = mappings["player_to_id"].get(batter, 0)
        bwl_id = mappings["player_to_id"].get(bowler, 0)

        # Build sequence feature array
        seq_features = []
        start_balls = max(1, team_balls - 19)
        runs_step = (current_runs / max(1, team_balls))

        for b in range(start_balls, team_balls + 1):
            r_left = max(0, target_runs - int(b * runs_step)) / max_runs_target
            b_norm = b / 120.0
            w_norm = team_wicket / 10.0
            seq_features.append([bat_id, bowl_id, ven_id, btr_id, bwl_id, r_left, b_norm, w_norm])

        # Pad sequence to 20 balls if needed
        while len(seq_features) < 20:
            seq_features.insert(0, [bat_id, bowl_id, ven_id, btr_id, bwl_id, target_runs / max_runs_target, 0.0, 0.0])

        x_input = torch.tensor(np.array([seq_features]), dtype=torch.float32)

        with torch.no_grad():
            win_prob = float(model(x_input).item())

        win_pct = win_prob * 100.0

        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Win Probability", f"{win_pct:.1f}%")
        with res_col2:
            st.metric("Runs Required", f"{runs_left} runs off {120 - team_balls} balls")
        with res_col3:
            req_rr = (runs_left / ((120 - team_balls) / 6.0)) if (120 - team_balls) > 0 else 0
            st.metric("Required Run Rate (RRR)", f"{req_rr:.2f}")

        st.progress(min(1.0, max(0.0, win_prob)))

        if win_prob >= 0.5:
            st.markdown(f"🏆 **Outcome Forecast:** <span class='win-badge'>{batting_team} is favored to WIN</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"⚡ **Outcome Forecast:** <span class='loss-badge'>{bowling_team} is favored to WIN</span>", unsafe_allow_html=True)

    # TAB 2: TEST DATASET TRAJECTORY
    elif app_mode == "📊 Test Dataset Trajectory":
        st.subheader("Test Dataset Sequence Trajectory")

        if data is None:
            st.warning("Processed test data not found.")
            return

        X_test = data["X_test"]
        y_test = data["y_test"]

        sample_idx = st.number_input("Select Test Sample Index:", min_value=0, max_value=len(X_test)-1, value=0)

        sample_seq = X_test[sample_idx]  # Shape: (20, 8)
        y_true = int(y_test[sample_idx])

        # Evaluate model probability across partial sequence windows (1 to 20 balls)
        probs_over_time = []
        with torch.no_grad():
            for t in range(1, 21):
                sub_seq = sample_seq[:t]
                if len(sub_seq) < 20:
                    pad = np.zeros((20 - len(sub_seq), sub_seq.shape[1]))
                    sub_seq = np.vstack([pad, sub_seq])
                sub_t = torch.tensor(np.array([sub_seq]), dtype=torch.float32)
                p = float(model(sub_t).item())
                probs_over_time.append(p * 100.0)

        final_prob = probs_over_time[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual Match Result", "WIN" if y_true == 1 else "LOSS")
        with col2:
            st.metric("Predicted Final Probability", f"{final_prob:.1f}%")

        st.subheader("Win Probability Trajectory Over 20-Ball Sequence Window")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 21), probs_over_time, marker="o", color="#10B981" if final_prob >= 50 else "#EF4444", linewidth=2.5)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% Threshold")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Ball Step in Window")
        ax.set_ylabel("Win Probability (%)")
        ax.set_title(f"Sample {sample_idx} Win Probability Trend")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # TAB 3: MODEL PERFORMANCE & INFO
    elif app_mode == "📈 Model Performance & Info":
        st.subheader("Model Performance & Architecture Summary")

        eval_path = "artifacts/evaluation_metrics.json"
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%")
            m2.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.4f}")
            m3.metric("Precision", f"{metrics.get('test_precision', 0):.4f}")
            m4.metric("F1 Score", f"{metrics.get('test_f1_score', 0):.4f}")

        st.markdown("### 🏗️ Network Architecture")
        st.code("""
LSTMWinPredictor(
  (lstm): LSTM(input_size=8, hidden_size=64, batch_first=True)
  (dropout): Dropout(p=0.2)
  (fc): Linear(in_features=64, out_features=1)
  (sigmoid): Sigmoid()
)
        """, language="python")

        hist_path = "artifacts/training_history.json"
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)

            st.markdown("### 📉 Training vs Validation Curves")
            epochs = range(1, len(history["train_loss"]) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3B82F6")
            ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#F59E0B")
            ax1.set_title("Loss Trajectory")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#10B981")
            ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#8B5CF6")
            ax2.set_title("Accuracy Trajectory")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)


if __name__ == "__main__":
    main()

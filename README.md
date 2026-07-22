# 🏏 Cricket Score & Win Probability Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A state-of-the-art Deep Learning pipeline for predicting Indian Premier League (IPL) **2nd Innings Win Probability** using Long Short-Term Memory (LSTM) recurrent neural networks.

---

## 📊 Statistical Overview & Empirical Performance

### 📈 Dataset Statistics (`IPL.csv`)

| Metric | Empirical Value | Description |
| :--- | :---: | :--- |
| **Total Ball-by-Ball Records** | **10,059** | Complete 2nd innings delivery records |
| **Total Matches** | **100** | Unique match IDs processed |
| **Sequence Length** | **20** | Sliding ball-by-ball window per sample |
| **Training Sequences (`X_train`)** | **6,545** | 80% train split sequence tensor shape `(6545, 20, 8)` |
| **Testing Sequences (`X_test`)** | **1,614** | 20% test split sequence tensor shape `(1614, 20, 8)` |
| **Unique Teams** | **8** | Franchise teams encoded |
| **Unique Venues** | **5** | Stadiums & grounds encoded |
| **Target Score Range** | **140 – 210** | 1st innings target score distribution (Mean: `174.25`) |

---

### 🎯 Model Evaluation Metrics

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Test Accuracy** | **53.66%** | Overall correct win/loss classifications |
| **ROC-AUC Score** | **0.6473** | Area Under Receiver Operating Characteristic Curve |
| **Precision** | **0.5725** | Positive prediction value |
| **Recall** | **0.6036** | True positive rate |
| **F1 Score** | **0.5877** | Harmonic mean of precision and recall |
| **Test Loss** | **1.7158** | Binary Cross-Entropy Loss |

---

## 🌟 Features

- **LSTM Sequence Modeling**: Captures ball-by-ball dynamic match trends over 20-ball sequence windows.
- **Kaggle IPL Dataset Integration**: Supports direct Kaggle downloading (`chaitu20/ipl-dataset2008-2025`) or local CSV ingestion.
- **Automated Data Processing**: Normalizes numeric match situation features and serializes team/player encodings.
- **Interactive Streamlit Web Dashboard**: Real-time match scenario simulator, 20-ball win probability trajectory graphs, and model metrics inspector.
- **Evaluation Suite**: Calculates Accuracy, ROC-AUC score, Loss, Precision, Recall, and F1 Score.
- **Cross-Platform & GPU Acceleration**: Supports CUDA, Apple MPS, and CPU backends.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Kaggle IPL Dataset / IPL.csv] --> B[data_preprocessing.py]
    B -->|Sequence Windows| C[artifacts/processed_data.npz]
    B -->|Encoder Mappings| D[artifacts/mappings.json]
    C --> E[train.py]
    E -->|Model Weights| F[artifacts/model.pt]
    C & F --> G[evaluate.py]
    C & F --> H[predict.py]
    D & F & C --> I[app.py Streamlit UI]
```

---

## 📂 Repository Structure

```text
.
├── IPL.csv                  # Kaggle IPL ball-by-ball dataset (root directory)
├── data_preprocessing.py    # Feature engineering, sliding-window sequence creation, mapping serialization
├── model.py                 # PyTorch LSTMWinPredictor architecture definition
├── train.py                 # Model training loop with validation & checkpointing
├── evaluate.py              # Test dataset metrics & evaluation report
├── predict.py               # Interactive CLI sample predictor with visual confidence bar
├── app.py                   # Streamlit web application (Match Simulator & Trajectory Plotter)
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore configuration
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/monal071/score-prediction.git
cd score-prediction
pip install -r requirements.txt
```

### 2. Preprocess Data

Place your downloaded Kaggle dataset as `IPL.csv` in the root folder, or download directly from Kaggle:

#### Option A: Local CSV (Default)
```bash
python data_preprocessing.py
```

#### Option B: Download Directly from Kaggle (`chaitu20/ipl-dataset2008-2025`)
```bash
pip install kagglehub
python data_preprocessing.py --kaggle chaitu20/ipl-dataset2008-2025
```

### 3. Train the LSTM Model

Trains the PyTorch model for 15 epochs and exports `artifacts/model.pt`:

```bash
python train.py
```

### 4. Evaluate Model Performance

Computes test accuracy, ROC-AUC score, precision, recall, and F1 score:

```bash
python evaluate.py
```

### 5. Run Single Sample Inference (CLI)

Predict win probability for a specific sample from the test set:

```bash
python predict.py --sample 0
```

---

## 💻 Launch Interactive Streamlit Dashboard

Run the interactive web app to simulate live match situations and plot win probability trajectories:

```bash
streamlit run app.py
```

Features in the Streamlit app:
- 🎯 **Match Scenario Simulator**: Adjust batting team, target runs, current score, balls bowled, and wickets lost to compute live win probability.
- 📊 **Test Dataset Trajectory**: Plot how win probability evolved ball-by-ball over any 20-ball sequence window in the test set.
- 📈 **Model Performance & Info**: View loss curves, ROC-AUC score, and neural network parameter summaries.

---

## 📊 Model Specifications

| Property | Description |
| :--- | :--- |
| **Model Type** | Recurrent Neural Network (LSTM) |
| **Sequence Length** | 20 consecutive balls |
| **Input Features (8)** | `batting_team`, `bowling_team`, `venue`, `batter`, `bowler`, `runs_left`, `team_balls`, `team_wicket` |
| **Hidden Dimensions** | 64 LSTM units |
| **Optimizer** | Adam (`lr=0.001`) |
| **Loss Function** | Binary Cross Entropy (`BCELoss`) |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.

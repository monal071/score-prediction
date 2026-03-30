# Cricket Score Prediction

This project predicts second-innings win probability using an LSTM model.

## Files

- `IPL.csv` : source dataset
- `data_preprocessing.py` : prepares train/test sequences and saves artifacts
- `model.py` : shared LSTM model class
- `train.py` : trains model and saves weights
- `evaluate.py` : evaluates model on test set
- `predict.py` : predicts one sample from saved test data
- `app.py` : local Streamlit app for interactive prediction
- `README.md` : project guide

## Setup

Install dependencies:

```bash
pip install numpy pandas torch scikit-learn streamlit
```

## Run Pipeline

1. Preprocess data

```bash
python data_preprocessing.py
```

2. Train model

```bash
python train.py
```

3. Evaluate model

```bash
python evaluate.py
```

4. Predict one sample

```bash
python predict.py
```

## Run Local App

```bash
streamlit run app.py
```

In the app, you can:

- Predict using an existing test sample index
- Upload a custom sequence CSV with 8 numeric features per ball

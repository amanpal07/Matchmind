# FootballIQ

FootballIQ predicts international football match outcomes using historical results, a custom Elo rating system, and two machine-learning classifiers.

## Features

- A custom Elo rating system with pre-match features
- Home win, draw, and away win probability estimates
- Logistic Regression and XGBoost models
- Models evaluated using a chronological holdout split
- Streamlit interface for match predictions, Elo rankings, and model comparison

## Dataset

Download the [International Football Results dataset from Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).

Place the CSV file named `results.csv` in the project root, alongside `train.py`.

The dataset is not included in this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train the models

```bash
python train.py
```

The training script evaluates Logistic Regression and XGBoost with and without Elo features, then fits deployment models on all available historical matches. Model files and metadata are saved in `artifacts/`.

## Run the Streamlit app

```bash
streamlit run app.py
```

The app requires the model artefacts created by the training step.

## Exploratory notebook results

In the reported experiment, Logistic Regression with Elo achieved the strongest macro-F1 score (0.492). XGBoost with Elo achieved the strongest accuracy (59.70%) and lowest log loss (0.879) among the reported configurations. These are results from the exploratory evaluation; retraining may produce different results depending on the data and split.

## Project limitations

The model uses historical team and tournament information. It does not incorporate live line-ups, injuries, suspensions, player-level statistics, or real-time form. Predicted probabilities are estimates, not guarantees, and this project is intended for learning and experimentation rather than betting advice.

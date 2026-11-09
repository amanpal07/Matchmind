# Matchmind

<https://footballmatchmind.streamlit.app/>

**Live application:** [Open Matchmind](https://footballmatchmind.streamlit.app/)

Matchmind predicts international football match outcomes using historical results, a custom Elo rating system, and two machine-learning classifiers.

## Features

- A custom Elo rating system with pre-match features
- Home win, draw, and away win probability estimates
- Logistic Regression and XGBoost models
- Models evaluated using a chronological holdout split
- Streamlit interface for match predictions, Elo rankings, and model comparison
- First-launch model training for deployment environments

## Dataset

The repository includes `results.csv`, the dataset supplied for this project. It contains 49,547 usable international matches, with the latest recorded match dated 26 August 2026.

The underlying dataset is the International Football Results collection by Mart Jürisoo (martj42), available on [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) and the maintainer's [GitHub repository](https://github.com/martj42/international_results). The Kaggle dataset is published under CC0.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train the models locally

```bash
python train.py
```

The training script evaluates Logistic Regression and XGBoost with and without Elo features using a chronological 80:20 holdout split. It then fits deployment models on all available historical matches and saves model files and metadata in `artifacts/`.

## Run the Streamlit app

```bash
streamlit run app.py
```

If model artefacts are not present, the app automatically runs the training pipeline once on first launch. This supports clean deployments such as Streamlit Community Cloud.

## Exploratory evaluation snapshot

On the supplied dataset, the chronological holdout evaluation reported:

| Model | Elo | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | No | 56.77% | 0.464 | 0.924 |
| Logistic Regression | Yes | 58.83% | 0.491 | 0.890 |
| XGBoost | No | 50.88% | 0.303 | 1.005 |
| XGBoost | Yes | 59.69% | 0.447 | 0.879 |

Accuracy and macro-F1 are higher-is-better; log loss is lower-is-better. These metrics describe this particular historical split and are not guarantees of future predictive performance.

## Project limitations

The model uses historical team and tournament information. It does not incorporate live line-ups, injuries, suspensions, player-level statistics, or real-time form. Predicted probabilities are estimates, not guarantees, and this project is intended for learning and experimentation rather than betting advice.

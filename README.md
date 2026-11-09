# FootballIQ

Football match outcome prediction using historical international football results, Elo ratings, and machine-learning models.

## Features
- Pre-match Elo ratings for home and away teams
- Logistic Regression and XGBoost classifiers
- Chronological evaluation to reduce temporal leakage
- Streamlit interface for match predictions, Elo rankings, and model comparison

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Download the Kaggle dataset [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) and place the CSV at `data/results.csv`.

Train the models and generate deployment artefacts:

```bash
python train.py --data data/results.csv
```

Launch the app:

```bash
streamlit run app.py
```

## Evaluation snapshot
The exploratory notebook evaluation found that Elo features improved the tested models. Logistic Regression with Elo gave the strongest macro-F1 (0.492), while XGBoost with Elo gave the strongest accuracy (59.70%) and log loss (0.879) among the reported experiments. Results depend on the evaluation split and dataset preparation.

## Notes
- The dataset is not included in this repository. Download it from Kaggle.
- Generated model artefacts and result files are created locally by the training script and are excluded from version control.
- This is an educational prediction project, not a betting or investment recommendation.

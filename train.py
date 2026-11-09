from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

DATA_PATH = Path("results.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

BASE_FEATURES = ["home_team", "away_team", "tournament", "year", "neutral"]
ELO_FEATURES = BASE_FEATURES + [
    "home_elo", "away_elo", "elo_difference", "expected_home_elo"
]
CATEGORICAL = ["home_team", "away_team", "tournament"]
LABELS = [0, 1, 2]
LABEL_NAMES = ["Away Win", "Draw", "Home Win"]


class EloRating:
    """Elo ratings with same-date matches using the same pre-date ratings."""

    def __init__(self, initial_rating=1500, K=32, home_advantage=100):
        self.initial_rating = initial_rating
        self.K = K
        self.home_advantage = home_advantage
        self.ratings = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def expected_score(self, home_elo, away_elo, neutral=False):
        advantage = 0 if bool(neutral) else self.home_advantage
        return 1 / (
            1 + 10 ** ((away_elo - home_elo - advantage) / 400)
        )

    def get_features(self, home, away, neutral=False):
        home_elo = self.get_rating(home)
        away_elo = self.get_rating(away)
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
            "expected_home_elo": self.expected_score(
                home_elo, away_elo, neutral
            ),
        }

    def fit_transform(self, df):
        data = df.sort_values("date", kind="stable").reset_index(drop=True).copy()
        records = []

        for _, matches in data.groupby("date", sort=True):
            updates = {}

            for _, row in matches.iterrows():
                home, away = row["home_team"], row["away_team"]
                neutral = bool(row["neutral"])

                # Features are captured before the match result is applied.
                features = self.get_features(home, away, neutral)
                records.append(features)

                actual_home = {0: 0.0, 1: 0.5, 2: 1.0}[int(row["result"])]
                change = self.K * (
                    actual_home - features["expected_home_elo"]
                )

                updates[home] = updates.get(home, 0.0) + change
                updates[away] = updates.get(away, 0.0) - change

            # Update after processing every match on this date.
            for team, change in updates.items():
                self.ratings[team] = self.get_rating(team) + change

        features_df = pd.DataFrame(records)
        return pd.concat([data, features_df], axis=1)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "results.csv not found. Put the Kaggle CSV in the project root."
        )

    df = pd.read_csv(DATA_PATH)
    required = [
        "date", "home_team", "away_team", "home_score",
        "away_score", "tournament", "neutral"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    neutral_map = {"true": 1, "false": 0, "1": 1, "0": 0}
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().map(neutral_map)
    )

    df = df.dropna(
        subset=[
            "date", "home_team", "away_team", "home_score",
            "away_score", "tournament", "neutral"
        ]
    ).copy()

    df["neutral"] = df["neutral"].astype(int)
    df["year"] = df["date"].dt.year

    # Same target encoding as the supplied notebook:
    # 0 = Away win, 1 = Draw, 2 = Home win.
    df["result"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] == df["away_score"],
        ],
        [2, 1],
        default=0,
    ).astype(int)

    return df.sort_values("date", kind="stable").reset_index(drop=True)


def make_preprocessor(use_elo, scale_numeric):
    numeric = ["year", "neutral"]
    if use_elo:
        numeric += [
            "home_elo", "away_elo", "elo_difference", "expected_home_elo"
        ]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
            (
                "numeric",
                StandardScaler() if scale_numeric else "passthrough",
                numeric,
            ),
        ]
    )


def make_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=5000, random_state=42)

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def make_pipeline(model_name, use_elo):
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    use_elo,
                    scale_numeric=(model_name == "Logistic Regression"),
                ),
            ),
            ("classifier", make_model(model_name)),
        ]
    )


def main():
    df = load_data()

    elo = EloRating(initial_rating=1500, K=32, home_advantage=100)
    df = elo.fit_transform(df)

    # Chronological 80:20 holdout, matching the notebook's approach.
    split_index = int(len(df) * 0.80)
    cutoff_date = df.iloc[split_index]["date"]
    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data for a chronological 80:20 split.")

    if set(train_df["result"].unique()) != set(LABELS):
        raise ValueError(
            "The training period must contain all three classes (0, 1, 2)."
        )

    metrics = {}
    print(f"Training matches: {len(train_df):,}")
    print(f"Testing matches:  {len(test_df):,}")
    print(f"Train ends: {train_df['date'].max().date()}")
    print(f"Test starts: {test_df['date'].min().date()}")

    # Evaluate both algorithms with and without Elo.
    for use_elo in (False, True):
        feature_columns = ELO_FEATURES if use_elo else BASE_FEATURES

        for model_name in ("Logistic Regression", "XGBoost"):
            pipeline = make_pipeline(model_name, use_elo)
            pipeline.fit(train_df[feature_columns], train_df["result"])

            predictions = pipeline.predict(test_df[feature_columns])
            probabilities = pipeline.predict_proba(test_df[feature_columns])

            key = f"{model_name} | Elo: {'Yes' if use_elo else 'No'}"
            metrics[key] = {
                "accuracy": float(
                    accuracy_score(test_df["result"], predictions)
                ),
                "macro_f1": float(
                    f1_score(
                        test_df["result"], predictions, average="macro"
                    )
                ),
                "log_loss": float(
                    log_loss(
                        test_df["result"], probabilities, labels=LABELS
                    )
                ),
                "classification_report": classification_report(
                    test_df["result"],
                    predictions,
                    labels=LABELS,
                    target_names=LABEL_NAMES,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    test_df["result"], predictions, labels=LABELS
                ).tolist(),
            }

    print("\nChronological holdout results:")
    for key, values in metrics.items():
        print(
            f"{key:36s} | "
            f"Accuracy {values['accuracy']:.4f} | "
            f"Macro-F1 {values['macro_f1']:.4f} | "
            f"Log-loss {values['log_loss']:.4f}"
        )

    # Refit deployable models on ALL historical matches after evaluation.
    # This avoids using a model trained only through the 2016 cutoff
    # alongside Elo ratings updated through the latest dataset date.
    deployed_models = {}
    for use_elo in (False, True):
        feature_columns = ELO_FEATURES if use_elo else BASE_FEATURES

        for model_name in ("Logistic Regression", "XGBoost"):
            pipeline = make_pipeline(model_name, use_elo)
            pipeline.fit(df[feature_columns], df["result"])
            deployed_models[(model_name, use_elo)] = pipeline

    joblib.dump(deployed_models, ARTIFACTS / "models.joblib")
    joblib.dump(
        {
            "ratings": elo.ratings,
            "initial_rating": elo.initial_rating,
            "K": elo.K,
            "home_advantage": elo.home_advantage,
        },
        ARTIFACTS / "elo_state.joblib",
    )

    metadata = {
        "teams": sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        ),
        "tournaments": sorted(df["tournament"].unique().tolist()),
        "last_historical_date": str(df["date"].max().date()),
        "match_count": int(len(df)),
        "cutoff_date": str(cutoff_date.date()),
        "evaluation": metrics,
    }
    joblib.dump(metadata, ARTIFACTS / "metadata.joblib")
    print("\nSaved models, Elo ratings and metadata in ./artifacts/")


if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import date
import joblib
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")
st.set_page_config(
    page_title="FootballIQ | Match Predictor",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .hero {
      padding: 1.6rem 1.8rem; border-radius: 18px;
      background: linear-gradient(115deg, #10233f, #174c70);
      color: white; margin-bottom: 1.2rem;
    }
    .hero p {color: #d6e5f5; margin-bottom: 0;}
    div[data-testid="stMetric"] {
      background: rgba(120,140,160,.08);
      border: 1px solid rgba(120,140,160,.2);
      padding: 12px; border-radius: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACTS / "models.joblib",
        ARTIFACTS / "elo_state.joblib",
        ARTIFACTS / "metadata.joblib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing)
            + ". Run `python train.py` after adding results.csv."
        )
    return (
        joblib.load(required[0]),
        joblib.load(required[1]),
        joblib.load(required[2]),
    )


try:
    models, elo_state, metadata = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()


def elo_for(team):
    return elo_state["ratings"].get(team, elo_state["initial_rating"])


def expected_home(home_elo, away_elo, neutral):
    advantage = 0 if neutral else elo_state["home_advantage"]
    return 1 / (
        1 + 10 ** ((away_elo - home_elo - advantage) / 400)
    )


def build_match(home, away, tournament, year, neutral, use_elo):
    row = {
        "home_team": home,
        "away_team": away,
        "tournament": tournament,
        "year": int(year),
        "neutral": int(neutral),
    }
    if use_elo:
        h, a = elo_for(home), elo_for(away)
        row.update(
            {
                "home_elo": h,
                "away_elo": a,
                "elo_difference": h - a,
                "expected_home_elo": expected_home(h, a, neutral),
            }
        )
    return pd.DataFrame([row])


st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">⚽ FootballIQ</h1>
      <h3 style="margin:.35rem 0;">International Football Match Predictor</h3>
      <p>Compare Logistic Regression and XGBoost, with optional Elo team-strength features.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_tab, ranking_tab, evaluation_tab, about_tab = st.tabs(
    ["⚽ Match Predictor", "🏆 Elo Rankings", "📊 Model Evaluation", "ℹ️ About"]
)

with predict_tab:
    controls, ratings = st.columns([1.15, 0.85], gap="large")

    with controls:
        st.subheader("Choose a fixture")
        teams = metadata["teams"]
        tournaments = metadata["tournaments"]

        home = st.selectbox("Home team", teams, key="home")
        away_list = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_list, key="away")
        tournament = st.selectbox("Tournament", tournaments)
        neutral = st.toggle("Neutral venue", value=False)
        match_year = st.number_input(
            "Match year",
            min_value=1872,
            max_value=2100,
            value=date.today().year,
            step=1,
        )
        algorithm = st.radio(
            "Algorithm",
            ["XGBoost", "Logistic Regression"],
            horizontal=True,
        )
        use_elo = st.toggle("Include Elo features", value=True)
        predict = st.button(
            "Predict match", type="primary", use_container_width=True
        )

    with ratings:
        st.subheader("Team strength")
        h_elo, a_elo = elo_for(home), elo_for(away)
        c1, c2 = st.columns(2)
        c1.metric("Home Elo", f"{h_elo:.0f}")
        c2.metric("Away Elo", f"{a_elo:.0f}")
        st.metric("Elo difference", f"{h_elo - a_elo:+.0f}")
        st.caption(
            f"Ratings calculated through {metadata['last_historical_date']}."
        )

    if predict:
        X = build_match(
            home, away, tournament, match_year, neutral, use_elo
        )
        model = models[(algorithm, use_elo)]
        probabilities = model.predict_proba(X)[0]
        classes = model.named_steps["classifier"].classes_
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        probs = {
            label_map[int(k)]: float(v)
            for k, v in zip(classes, probabilities)
        }

        st.divider()
        st.subheader(f"{home} vs {away}")
        outcome = max(probs, key=probs.get)
        st.success(f"Most likely outcome: **{outcome}**")

        a, b, c = st.columns(3)
        a.metric("Home win", f"{probs.get('Home Win', 0):.1%}")
        b.metric("Draw", f"{probs.get('Draw', 0):.1%}")
        c.metric("Away win", f"{probs.get('Away Win', 0):.1%}")

        chart = pd.DataFrame(
            {"Probability": [
                probs.get("Home Win", 0),
                probs.get("Draw", 0),
                probs.get("Away Win", 0),
            ]},
            index=["Home Win", "Draw", "Away Win"],
        )
        st.bar_chart(chart)
        st.caption(
            f"{algorithm} · Elo features: {'On' if use_elo else 'Off'}"
        )
        st.info(
            "Probabilities are model estimates, not guarantees. "
            "The model does not include live injuries, line-ups, suspensions "
            "or real-time form."
        )


with ranking_tab:
    st.subheader("Historical Elo rankings")
    ranking = pd.DataFrame(
        elo_state["ratings"].items(),
        columns=["Team", "Elo Rating"],
    ).sort_values("Elo Rating", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "Rank"
    query = st.text_input("Filter teams")
    if query:
        ranking = ranking[
            ranking["Team"].str.contains(query, case=False, na=False)
        ]
    st.dataframe(
        ranking.style.format({"Elo Rating": "{:.1f}"}),
        use_container_width=True,
        height=550,
    )


with evaluation_tab:
    st.subheader("Chronological holdout evaluation")
    rows = []
    for key, values in metadata["evaluation"].items():
        model_name, elo_label = key.split(" | Elo: ")
        rows.append(
            {
                "Model": model_name,
                "Elo": elo_label,
                "Accuracy": values["accuracy"],
                "Macro F1": values["macro_f1"],
                "Log Loss": values["log_loss"],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "Accuracy": "{:.3f}",
                "Macro F1": "{:.3f}",
                "Log Loss": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Chronological split: training before {metadata['cutoff_date']}; "
        f"testing from {metadata['cutoff_date']} onwards."
    )
    st.markdown("**Accuracy (higher is better)**")
    st.bar_chart(table.set_index(["Model", "Elo"])["Accuracy"])
    st.markdown("**Log loss (lower is better)**")
    st.bar_chart(table.set_index(["Model", "Elo"])["Log Loss"])


with about_tab:
    st.subheader("About FootballIQ")
    st.write(
        "A football outcome prediction project using international match "
        "history, a custom Elo rating system, Logistic Regression and XGBoost."
    )
    st.write(f"Historical matches: **{metadata['match_count']:,}**")
    st.write(f"Latest match in dataset: **{metadata['last_historical_date']}**")
    st.write("Outcome encoding: 0 = Away Win, 1 = Draw, 2 = Home Win.")
    st.warning(
        "The model is based on historical team and tournament information. "
        "It does not use live player availability or current match-day data."
    )

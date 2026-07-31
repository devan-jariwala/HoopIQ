from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset_v2.csv"
MODELS_PATH = PROJECT_ROOT / "models"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["GAME_DATE"])


@st.cache_resource
def load_models():
    points_model = joblib.load(MODELS_PATH / "points_model.pkl")
    rebounds_model = joblib.load(MODELS_PATH / "rebounds_model.pkl")
    assists_model = joblib.load(MODELS_PATH / "assists_model.pkl")
    feature_columns = joblib.load(MODELS_PATH / "feature_columns.pkl")

    return points_model, rebounds_model, assists_model, feature_columns


data = load_data()

(
    points_model,
    rebounds_model,
    assists_model,
    feature_columns,
) = load_models()


st.set_page_config(
    page_title="HoopIQ",
    page_icon="🏀",
    layout="centered",
)

st.title("🏀 HoopIQ")
st.subheader("NBA Next-Game Performance Predictor")

st.write(
    "Generate a projected next-game stat line using a player's latest "
    "historical performance, opportunity, and consistency profile."
)


player_names = sorted(data["PLAYER_NAME"].dropna().unique())

selected_player = st.selectbox(
    "Select a player",
    player_names,
)

game_location = st.radio(
    "Game location",
    ["Home", "Away"],
    horizontal=True,
)

rest_days = st.number_input(
    "Days since previous game",
    min_value=1,
    max_value=30,
    value=2,
    step=1,
)


player_data = (
    data[data["PLAYER_NAME"] == selected_player]
    .sort_values("GAME_DATE")
)

latest_row = player_data.iloc[-1].copy()

# Override the controllable pre-game context
latest_row["IS_HOME"] = 1 if game_location == "Home" else 0
latest_row["REST_DAYS"] = rest_days

model_input = latest_row[feature_columns].to_frame().T


if st.button("Generate Prediction", type="primary"):
    predicted_points = points_model.predict(model_input)[0]
    predicted_rebounds = rebounds_model.predict(model_input)[0]
    predicted_assists = assists_model.predict(model_input)[0]

    st.divider()
    st.subheader(f"{selected_player} projected stat line")

    col1, col2, col3 = st.columns(3)

    col1.metric("Points", f"{predicted_points:.1f}")
    col2.metric("Rebounds", f"{predicted_rebounds:.1f}")
    col3.metric("Assists", f"{predicted_assists:.1f}")

    st.caption(
        f"Scenario: {game_location.lower()} game with {rest_days} day(s) "
        "since the player's previous game."
    )

    st.subheader("Recent player profile")

    context = pd.DataFrame({
        "Metric": [
            "Last 10 points",
            "Last 10 rebounds",
            "Last 10 assists",
            "Last 10 minutes",
            "Last 10 field-goal attempts",
            "Points consistency",
        ],
        "Value": [
            latest_row["PTS_LAST_10_AVG"],
            latest_row["REB_LAST_10_AVG"],
            latest_row["AST_LAST_10_AVG"],
            latest_row["MIN_LAST_10_AVG"],
            latest_row["FGA_LAST_10_AVG"],
            latest_row["PTS_STD_LAST_10"],
        ],
    })

    context["Value"] = context["Value"].round(1)

    st.dataframe(
        context,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "This model predicts a hypothetical next game using the player's "
        "latest available 2025–26 profile. Opponent-specific defensive and "
        "matchup history features are not yet included."
    )
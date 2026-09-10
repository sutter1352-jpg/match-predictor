import pandas as pd
import streamlit as st

from src.match_predictor import load_matches
from src.match_stats_predictor import resolve_stat_columns
from src.daily_match_stats_simulator import (
    get_all_fixtures,
    simulate_fixture,
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Big Five Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    .match-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 22px;
    }

    .league-name {
        font-size: 0.85rem;
        font-weight: 700;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
    }

    .match-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .prediction {
        font-size: 1rem;
        margin-bottom: 16px;
    }

    .score-box {
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        background: rgba(128,128,128,0.08);
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .score-number {
        font-size: 2rem;
        font-weight: 800;
    }

    .small-muted {
        font-size: 0.82rem;
        opacity: 0.65;
    }

    @media (max-width: 600px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 1.8rem;
        }

        .match-title {
            font-size: 1.2rem;
        }

        .match-card {
            padding: 16px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_model_data():

    matches = load_matches()

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce"
    )

    matches = (
        matches
        .dropna(
            subset=[
                "date",
                "league_code",
                "home_team",
                "away_team",
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    stat_columns = resolve_stat_columns(
        matches
    )

    return (
        matches,
        stat_columns,
    )


# ============================================================
# FETCH FIXTURES
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def fetch_fixtures(
    date_string
):

    prediction_date = pd.Timestamp(
        date_string
    )

    return get_all_fixtures(
        prediction_date
    )


# ============================================================
# RUN ALL PREDICTIONS
# ============================================================

def run_predictions(
    matches,
    stat_columns,
    prediction_date
):

    fixtures = fetch_fixtures(
        str(
            prediction_date.date()
        )
    )

    predictions = []
    failures = []

    if not fixtures:

        return (
            predictions,
            failures,
            fixtures,
        )

    progress = st.progress(0)

    status_box = st.empty()

    total = len(fixtures)

    for index, fixture in enumerate(
        fixtures,
        start=1
    ):

        status_box.write(
            f"Analyzing "
            f"{fixture['home_external']} "
            f"vs "
            f"{fixture['away_external']}..."
        )

        try:

            prediction = simulate_fixture(
                fixture,
                matches,
                stat_columns,
                prediction_date
            )

            if prediction is None:

                failures.append(
                    fixture
                )

            else:

                predictions.append(
                    prediction
                )

        except Exception as error:

            fixture_copy = dict(
                fixture
            )

            fixture_copy["error"] = str(
                error
            )

            failures.append(
                fixture_copy
            )

        progress.progress(
            index / total
        )

    status_box.empty()

    progress.empty()

    return (
        predictions,
        failures,
        fixtures,
    )


# ============================================================
# PROBABILITY BAR
# ============================================================

def probability_bar(
    label,
    probability
):

    st.write(
        f"**{label}** — "
        f"{probability:.1%}"
    )

    st.progress(
        min(
            max(
                float(probability),
                0.0
            ),
            1.0
        )
    )


# ============================================================
# MATCH CARD
# ============================================================

def display_match_card(
    match
):

    home = match[
        "home_team"
    ]

    away = match[
        "away_team"
    ]

    st.markdown(
        f"""
        <div class="match-card">

            <div class="league-name">
                {match['league']}
            </div>

            <div class="match-title">
                {home} vs {away}
            </div>

            <div class="prediction">
                Prediction:
                <strong>
                    {match['predicted_result']}
                </strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # RESULT PROBABILITIES
    # ========================================================

    st.markdown(
        "#### Match Result"
    )

    probability_bar(
        home,
        match[
            "home_probability"
        ]
    )

    probability_bar(
        "Draw",
        match[
            "draw_probability"
        ]
    )

    probability_bar(
        away,
        match[
            "away_probability"
        ]
    )

    # ========================================================
    # SCORE
    # ========================================================

    st.markdown(
        f"""
        <div class="score-box">

            <div class="small-muted">
                MOST LIKELY SCORE
            </div>

            <div class="score-number">
                {home}
                {match['predicted_home_goals']}
                -
                {match['predicted_away_goals']}
                {away}
            </div>

            <div class="small-muted">
                Exact score probability:
                {match['score_probability']:.1%}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # STATS
    # ========================================================

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            "##### Expected Goals"
        )

        st.metric(
            home,
            f"{match['home_xg']:.2f}"
        )

        st.metric(
            away,
            f"{match['away_xg']:.2f}"
        )

        st.caption(
            f"Total: "
            f"{match['total_xg']:.2f}"
        )

    with col2:

        st.markdown(
            "##### Shots on Target"
        )

        st.metric(
            home,
            f"{match['home_sot']:.2f}"
        )

        st.metric(
            away,
            f"{match['away_sot']:.2f}"
        )

        st.caption(
            f"Total: "
            f"{match['total_sot']:.2f}"
        )

    with col3:

        st.markdown(
            "##### Corners"
        )

        st.metric(
            home,
            f"{match['home_corners']:.2f}"
        )

        st.metric(
            away,
            f"{match['away_corners']:.2f}"
        )

        st.caption(
            f"Total: "
            f"{match['total_corners']:.2f}"
        )

    status = match.get(
        "status"
    )

    if status:

        st.caption(
            f"Fixture status: {status}"
        )

    st.divider()


# ============================================================
# SUMMARY TABLE
# ============================================================

def create_summary_table(
    predictions
):

    rows = []

    for match in predictions:

        rows.append({

            "League":
                match[
                    "league"
                ],

            "Match":
                (
                    f"{match['home_team']} "
                    f"vs "
                    f"{match['away_team']}"
                ),

            "Prediction":
                match[
                    "predicted_result"
                ],

            "Home %":
                round(
                    match[
                        "home_probability"
                    ]
                    * 100,
                    1
                ),

            "Draw %":
                round(
                    match[
                        "draw_probability"
                    ]
                    * 100,
                    1
                ),

            "Away %":
                round(
                    match[
                        "away_probability"
                    ]
                    * 100,
                    1
                ),

            "Score":
                match[
                    "predicted_score"
                ],

            "xG":
                (
                    f"{match['home_xg']:.1f}"
                    "-"
                    f"{match['away_xg']:.1f}"
                ),

            "SOT":
                (
                    f"{match['home_sot']:.1f}"
                    "-"
                    f"{match['away_sot']:.1f}"
                ),

            "Corners":
                (
                    f"{match['home_corners']:.1f}"
                    "-"
                    f"{match['away_corners']:.1f}"
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ⚽ Big Five Match Predictor
    </div>

    <div class="subtitle">
        Result probabilities, goals, shots on target and corners
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    matches, stat_columns = (
        load_model_data()
    )

except Exception as error:

    st.error(
        "Could not load the prediction model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# CONTROLS
# ============================================================

control1, control2 = st.columns(
    [
        2,
        1,
    ]
)

with control1:

    selected_date = st.date_input(
        "Match date",
        value=pd.Timestamp.today().date(),
    )

with control2:

    st.write("")

    st.write("")

    run_button = st.button(
        "⚽ Run Predictions",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# LEAGUE FILTER
# ============================================================

league_options = [
    "All Leagues",
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
]

selected_league = st.selectbox(
    "League",
    league_options,
)


# ============================================================
# SESSION STORAGE
# ============================================================

if (
    "predictions"
    not in st.session_state
):

    st.session_state[
        "predictions"
    ] = None


if (
    "prediction_date"
    not in st.session_state
):

    st.session_state[
        "prediction_date"
    ] = None


if (
    "failures"
    not in st.session_state
):

    st.session_state[
        "failures"
    ] = []


# ============================================================
# RUN MODEL
# ============================================================

if run_button:

    prediction_date = pd.Timestamp(
        selected_date
    )

    with st.spinner(
        "Finding fixtures and running models..."
    ):

        (
            predictions,
            failures,
            fixtures,
        ) = run_predictions(
            matches,
            stat_columns,
            prediction_date
        )

    st.session_state[
        "predictions"
    ] = predictions

    st.session_state[
        "prediction_date"
    ] = prediction_date

    st.session_state[
        "failures"
    ] = failures

    if not fixtures:

        st.warning(
            "No Big Five fixtures were "
            "found for that date."
        )

    elif not predictions:

        st.error(
            "Fixtures were found, but none "
            "could be modeled."
        )

    else:

        st.success(
            f"{len(predictions)} matches predicted."
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

predictions = (
    st.session_state[
        "predictions"
    ]
)

if predictions:

    # ========================================================
    # FILTER
    # ========================================================

    if (
        selected_league
        != "All Leagues"
    ):

        visible_predictions = [
            match
            for match in predictions
            if (
                match[
                    "league"
                ]
                ==
                selected_league
            )
        ]

    else:

        visible_predictions = (
            predictions
        )

    # ========================================================
    # DATE
    # ========================================================

    prediction_date = (
        st.session_state[
            "prediction_date"
        ]
    )

    st.markdown(
        f"### Predictions for "
        f"{prediction_date.strftime('%B %d, %Y')}"
    )

    st.caption(
        f"{len(visible_predictions)} "
        f"matches shown"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if visible_predictions:

        st.markdown(
            "### Daily Summary"
        )

        summary = create_summary_table(
            visible_predictions
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # CSV DOWNLOAD
        # ====================================================

        full_csv = pd.DataFrame(
            visible_predictions
        ).to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "Download predictions CSV",
            data=full_csv,
            file_name=(
                "predictions_"
                f"{prediction_date.date()}"
                ".csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        # ====================================================
        # MATCH CARDS
        # ====================================================

        st.markdown(
            "### Match Details"
        )

        for match in (
            visible_predictions
        ):

            display_match_card(
                match
            )

    else:

        st.info(
            "No matches from the selected "
            "league on this date."
        )


# ============================================================
# FAILED FIXTURES
# ============================================================

failures = (
    st.session_state[
        "failures"
    ]
)

if failures:

    with st.expander(
        f"⚠️ {len(failures)} fixtures "
        f"could not be modeled"
    ):

        for fixture in failures:

            st.write(
                f"**{fixture.get('league', '')}:** "
                f"{fixture.get('home_external', '?')} "
                f"vs "
                f"{fixture.get('away_external', '?')}"
            )

            if fixture.get(
                "error"
            ):

                st.caption(
                    fixture[
                        "error"
                    ]
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictions are statistical model estimates. "
    "Shots on target and corner models are experimental "
    "and should be validated with historical backtesting."
)
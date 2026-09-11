import html

import pandas as pd
import requests
import streamlit as st

from src.config import LEAGUES

from src.match_predictor import (
    load_matches,
)

from src.match_stats_predictor import (
    resolve_stat_columns,
)

from src.daily_match_stats_simulator import (
    get_all_fixtures,
    simulate_fixture,
    ESPN_LEAGUES,
)

from src.live_data_updater import (
    refresh_current_season,
)


# ============================================================
# SETTINGS
# ============================================================

LIVE_REFRESH_SECONDS = 30
REQUEST_TIMEOUT = 20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Big Five Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    .score-box {
        text-align: center;
        background: rgba(128,128,128,0.08);
        border-radius: 14px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .score-number {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .small-muted {
        opacity: 0.65;
        font-size: 0.82rem;
    }

    .live-game {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .live-league {
        opacity: 0.60;
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05rem;
    }

    .live-match-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-top: 5px;
    }

    .live-teams {
        font-size: 1rem;
        font-weight: 700;
        min-width: 0;
    }

    .live-score {
        font-size: 1.25rem;
        font-weight: 800;
        white-space: nowrap;
    }

    .live-status {
        margin-top: 5px;
        font-size: 0.80rem;
        opacity: 0.75;
    }

    .update-box {
        border-radius: 12px;
        background: rgba(128,128,128,0.08);
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 16px;
    }

    @media (max-width: 600px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 1.8rem;
        }

        .live-match-row {
            align-items: flex-start;
        }

        .live-score {
            font-size: 1.1rem;
        }

        .score-number {
            font-size: 1.35rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC DATA CLEANUP
# ============================================================

def clean_matches(
    matches
):

    matches = matches.copy()

    matches[
        "date"
    ] = pd.to_datetime(
        matches[
            "date"
        ],
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
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    return matches


# ============================================================
# LOAD BASE DATABASE
#
# Cached briefly.
# When Run Predictions is pressed we clear the cache first.
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def load_base_matches():

    matches = load_matches()

    return clean_matches(
        matches
    )


# ============================================================
# ESPN SCORE HELPERS
# ============================================================

def get_espn_score(
    competitor
):

    score = competitor.get(
        "score"
    )

    if isinstance(
        score,
        dict
    ):

        score = (
            score.get(
                "displayValue"
            )
            or
            score.get(
                "value"
            )
        )

    if score is None:
        return None

    return str(
        score
    )


# ============================================================
# GET LIVE / FINAL / SCHEDULED SCORES
# ============================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def fetch_live_scores(
    date_string
):

    date_value = (
        pd.Timestamp(
            date_string
        )
        .strftime(
            "%Y%m%d"
        )
    )

    games = []
    errors = []

    for (
        league_code,
        espn_code,
    ) in ESPN_LEAGUES.items():

        url = (
            "https://site.api.espn.com/"
            "apis/site/v2/sports/soccer/"
            f"{espn_code}/scoreboard"
        )

        try:

            response = requests.get(
                url,
                params={
                    "dates":
                        date_value,

                    "limit":
                        100,
                },
                timeout=
                    REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = (
                response.json()
            )

        except Exception as error:

            errors.append(
                (
                    LEAGUES[
                        league_code
                    ],
                    str(error),
                )
            )

            continue

        events = (
            data.get(
                "events"
            )
            or []
        )

        for event in events:

            competitions = (
                event.get(
                    "competitions"
                )
                or []
            )

            if not competitions:
                continue

            competition = (
                competitions[0]
            )

            competitors = (
                competition.get(
                    "competitors"
                )
                or []
            )

            home_team = None
            away_team = None

            home_score = None
            away_score = None

            for competitor in competitors:

                team_data = (
                    competitor.get(
                        "team"
                    )
                    or {}
                )

                team_name = (
                    team_data.get(
                        "displayName"
                    )
                    or
                    team_data.get(
                        "shortDisplayName"
                    )
                    or
                    team_data.get(
                        "name"
                    )
                    or
                    "Unknown"
                )

                side = (
                    competitor.get(
                        "homeAway"
                    )
                )

                score = get_espn_score(
                    competitor
                )

                if side == "home":

                    home_team = (
                        team_name
                    )

                    home_score = (
                        score
                    )

                elif side == "away":

                    away_team = (
                        team_name
                    )

                    away_score = (
                        score
                    )

            if (
                not home_team
                or not away_team
            ):

                continue

            status_type = (
                event.get(
                    "status",
                    {}
                )
                .get(
                    "type",
                    {}
                )
            )

            state = (
                status_type.get(
                    "state"
                )
                or ""
            ).lower()

            completed = bool(
                status_type.get(
                    "completed",
                    False
                )
            )

            detail = (
                status_type.get(
                    "shortDetail"
                )
                or
                status_type.get(
                    "detail"
                )
                or
                status_type.get(
                    "description"
                )
                or
                status_type.get(
                    "name"
                )
                or
                ""
            )

            if (
                state == "in"
            ):

                status_group = "live"

            elif (
                state == "post"
                or completed
            ):

                status_group = "final"

            else:

                status_group = "scheduled"

            games.append({

                "event_id":
                    str(
                        event.get(
                            "id",
                            ""
                        )
                    ),

                "league_code":
                    league_code,

                "league":
                    LEAGUES[
                        league_code
                    ],

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "home_score":
                    home_score,

                "away_score":
                    away_score,

                "status":
                    detail,

                "status_group":
                    status_group,

                "kickoff":
                    event.get(
                        "date"
                    ),
            })

    # Live games first, then scheduled,
    # then completed.
    priority = {
        "live": 0,
        "scheduled": 1,
        "final": 2,
    }

    games.sort(
        key=lambda game: (
            priority.get(
                game[
                    "status_group"
                ],
                9
            ),
            game[
                "league"
            ],
            str(
                game.get(
                    "kickoff",
                    ""
                )
            ),
        )
    )

    return (
        games,
        errors,
    )


# ============================================================
# LIVE SCORE CARD
# ============================================================

def display_live_game(
    game
):

    home = html.escape(
        str(
            game[
                "home_team"
            ]
        )
    )

    away = html.escape(
        str(
            game[
                "away_team"
            ]
        )
    )

    league = html.escape(
        str(
            game[
                "league"
            ]
        )
    )

    status = html.escape(
        str(
            game.get(
                "status",
                ""
            )
        )
    )

    status_group = (
        game[
            "status_group"
        ]
    )

    if status_group == "live":

        status_text = (
            f"🔴 LIVE · {status}"
            if status
            else
            "🔴 LIVE"
        )

    elif status_group == "final":

        status_text = (
            "✅ Final"
        )

    else:

        status_text = (
            f"🕒 {status}"
            if status
            else
            "🕒 Scheduled"
        )

    if (
        status_group
        == "scheduled"
    ):

        score_text = "vs"

    else:

        home_score = (
            game[
                "home_score"
            ]
            if game[
                "home_score"
            ]
            is not None
            else "-"
        )

        away_score = (
            game[
                "away_score"
            ]
            if game[
                "away_score"
            ]
            is not None
            else "-"
        )

        score_text = (
            f"{home_score} - "
            f"{away_score}"
        )

    st.markdown(
        f"""
        <div class="live-game">

            <div class="live-league">
                {league}
            </div>

            <div class="live-match-row">

                <div class="live-teams">
                    {home}<br>
                    {away}
                </div>

                <div class="live-score">
                    {score_text}
                </div>

            </div>

            <div class="live-status">
                {status_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LIVE SCORE CENTER
#
# Only this section reruns every 30 seconds.
# ============================================================

@st.fragment(
    run_every=
        LIVE_REFRESH_SECONDS
)
def live_score_center(
    selected_date,
    selected_league
):

    header1, header2 = (
        st.columns(
            [
                3,
                1,
            ]
        )
    )

    with header1:

        st.markdown(
            "### 🔴 Live Score Center"
        )

        st.caption(
            "Scores refresh automatically "
            f"every {LIVE_REFRESH_SECONDS} seconds "
            "while this page is open."
        )

    with header2:

        refresh_now = (
            st.button(
                "↻ Refresh",
                use_container_width=True,
                key=(
                    "live_refresh_"
                    f"{selected_date}_"
                    f"{selected_league}"
                ),
            )
        )

    if refresh_now:

        fetch_live_scores.clear()

    games, errors = (
        fetch_live_scores(
            str(
                selected_date
            )
        )
    )

    if (
        selected_league
        != "All Leagues"
    ):

        games = [
            game
            for game in games
            if (
                game[
                    "league"
                ]
                ==
                selected_league
            )
        ]

    if not games:

        st.info(
            "No games found for this "
            "league/date."
        )

    else:

        live_count = sum(
            1
            for game in games
            if (
                game[
                    "status_group"
                ]
                == "live"
            )
        )

        if live_count:

            st.caption(
                f"{live_count} match"
                f"{'es' if live_count != 1 else ''} "
                "currently live."
            )

        for game in games:

            display_live_game(
                game
            )

    if errors:

        with st.expander(
            "Live score connection warnings"
        ):

            for (
                league,
                error,
            ) in errors:

                st.write(
                    f"**{league}:** "
                    f"{error}"
                )


# ============================================================
# RUN MATCH PREDICTIONS
# ============================================================

def run_predictions(
    matches,
    stat_columns,
    prediction_date
):

    fixtures = get_all_fixtures(
        prediction_date
    )

    predictions = []
    failures = []

    if not fixtures:

        return (
            predictions,
            failures,
            fixtures,
        )

    progress = st.progress(
        0
    )

    status_box = st.empty()

    total = len(
        fixtures
    )

    for (
        index,
        fixture,
    ) in enumerate(
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

            prediction = (
                simulate_fixture(
                    fixture,
                    matches,
                    stat_columns,
                    prediction_date
                )
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

            failed_fixture = (
                dict(
                    fixture
                )
            )

            failed_fixture[
                "error"
            ] = str(
                error
            )

            failures.append(
                failed_fixture
            )

        progress.progress(
            index / total
        )

    progress.empty()
    status_box.empty()

    return (
        predictions,
        failures,
        fixtures,
    )


# ============================================================
# MATCH PREDICTION CARD
# ============================================================

def display_prediction_card(
    match
):

    home = (
        match[
            "home_team"
        ]
    )

    away = (
        match[
            "away_team"
        ]
    )

    with st.container(
        border=True
    ):

        st.caption(
            match[
                "league"
            ]
        )

        st.subheader(
            f"{home} vs {away}"
        )

        st.markdown(
            f"**Most likely outcome:** "
            f"{match['predicted_result']}"
        )

        # ====================================================
        # RESULT PROBABILITIES
        # ====================================================

        st.markdown(
            "##### Match Result"
        )

        result1, result2, result3 = (
            st.columns(
                3
            )
        )

        with result1:

            st.metric(
                home,
                f"{match['home_probability']:.1%}"
            )

        with result2:

            st.metric(
                "Draw",
                f"{match['draw_probability']:.1%}"
            )

        with result3:

            st.metric(
                away,
                f"{match['away_probability']:.1%}"
            )

        # ====================================================
        # EXACT SCORE
        # ====================================================

        safe_home = html.escape(
            home
        )

        safe_away = html.escape(
            away
        )

        st.markdown(
            f"""
            <div class="score-box">

                <div class="small-muted">
                    MOST LIKELY EXACT SCORE
                </div>

                <div class="score-number">
                    {safe_home}
                    {match['predicted_home_goals']}
                    -
                    {match['predicted_away_goals']}
                    {safe_away}
                </div>

                <div class="small-muted">
                    Probability of this exact score:
                    {match['score_probability']:.1%}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # STATS
        # ====================================================

        stat1, stat2, stat3 = (
            st.columns(
                3
            )
        )

        with stat1:

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

        with stat2:

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

        with stat3:

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
# SESSION STATE
# ============================================================

defaults = {

    "predictions":
        None,

    "prediction_date":
        None,

    "failures":
        [],

    "update_report":
        None,

    "model_matches_count":
        None,

    "model_latest_date":
        None,
}

for (
    key,
    value,
) in defaults.items():

    if (
        key
        not in st.session_state
    ):

        st.session_state[
            key
        ] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ⚽ Big Five Match Predictor
    </div>

    <div class="subtitle">
        Result probabilities, goals, shots on target,
        corners and live scores
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATE / LEAGUE CONTROLS
# ============================================================

control1, control2 = (
    st.columns(
        2
    )
)

with control1:

    selected_date = (
        st.date_input(
            "Match date",
            value=
                pd.Timestamp
                .today()
                .date(),
        )
    )

with control2:

    league_options = [
        "All Leagues",
        *list(
            LEAGUES.values()
        ),
    ]

    selected_league = (
        st.selectbox(
            "League",
            league_options,
        )
    )


# ============================================================
# LIVE SCORES
# ============================================================

live_score_center(
    selected_date,
    selected_league
)


# ============================================================
# DIVIDER
# ============================================================

st.divider()


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    "## 🔮 Pre-Match Predictions"
)

st.caption(
    "Before every prediction run, the app checks "
    "for newly completed Big Five matches and "
    "incorporates them into the model data."
)


run_button = st.button(
    "⚽ Update Data & Run Predictions",
    type="primary",
    use_container_width=True,
)


# ============================================================
# UPDATE DATABASE + RUN MODEL
# ============================================================

if run_button:

    prediction_date = (
        pd.Timestamp(
            selected_date
        ).normalize()
    )

    # ========================================================
    # FORCE FRESH BASE LOAD
    # ========================================================

    load_base_matches.clear()

    with st.spinner(
        "Loading historical database..."
    ):

        base_matches = (
            load_base_matches()
        )

    # ========================================================
    # CHECK FOR NEW COMPLETED MATCHES
    # ========================================================

    with st.spinner(
        "Checking all five leagues for "
        "newly completed matches..."
    ):

        try:

            (
                updated_matches,
                update_report,
            ) = refresh_current_season(
                base_matches,
                save=False,
                verbose=False
            )

            updated_matches = (
                clean_matches(
                    updated_matches
                )
            )

        except Exception as error:

            updated_matches = (
                base_matches
            )

            update_report = {

                "new_matches":
                    0,

                "refreshed_matches":
                    0,

                "errors": {
                    "Updater":
                        str(error)
                },
            }

    # ========================================================
    # STAT COLUMNS
    # ========================================================

    try:

        stat_columns = (
            resolve_stat_columns(
                updated_matches
            )
        )

    except Exception as error:

        st.error(
            "Could not find the shots-on-target "
            "or corner columns."
        )

        st.exception(
            error
        )

        st.stop()

    # ========================================================
    # STORE UPDATE INFORMATION
    # ========================================================

    st.session_state[
        "update_report"
    ] = update_report

    st.session_state[
        "model_matches_count"
    ] = len(
        updated_matches
    )

    latest_date = (
        updated_matches[
            "date"
        ].max()
    )

    if pd.notna(
        latest_date
    ):

        st.session_state[
            "model_latest_date"
        ] = latest_date.date()

    # ========================================================
    # RUN PREDICTIONS
    # ========================================================

    with st.spinner(
        "Finding fixtures and running predictions..."
    ):

        (
            predictions,
            failures,
            fixtures,
        ) = run_predictions(
            updated_matches,
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

    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if not fixtures:

        st.warning(
            "No Big Five fixtures were "
            "found for this date."
        )

    elif not predictions:

        st.error(
            "Fixtures were found, but none "
            "could be modeled."
        )

    else:

        st.success(
            f"{len(predictions)} "
            "matches predicted successfully."
        )


# ============================================================
# DATABASE UPDATE STATUS
# ============================================================

update_report = (
    st.session_state[
        "update_report"
    ]
)

if update_report is not None:

    new_matches = (
        update_report.get(
            "new_matches",
            0
        )
    )

    refreshed = (
        update_report.get(
            "refreshed_matches",
            0
        )
    )

    match_count = (
        st.session_state[
            "model_matches_count"
        ]
    )

    latest_date = (
        st.session_state[
            "model_latest_date"
        ]
    )

    st.markdown(
        f"""
        <div class="update-box">

        <strong>📚 Model data updated</strong><br>

        New completed matches found:
        <strong>{new_matches}</strong><br>

        Existing current-season matches refreshed:
        <strong>{refreshed}</strong><br>

        Matches available to model:
        <strong>{match_count:,}</strong><br>

        Latest match in model:
        <strong>{latest_date}</strong>

        </div>
        """,
        unsafe_allow_html=True,
    )

    errors = (
        update_report.get(
            "errors",
            {}
        )
    )

    if errors:

        with st.expander(
            "⚠️ Data update warnings"
        ):

            for (
                league,
                error,
            ) in errors.items():

                st.write(
                    f"**{league}:** "
                    f"{error}"
                )


# ============================================================
# PREDICTIONS
# ============================================================

predictions = (
    st.session_state[
        "predictions"
    ]
)

if predictions:

    # ========================================================
    # LEAGUE FILTER
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
        "matches shown"
    )

    if visible_predictions:

        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            "### Daily Summary"
        )

        summary = (
            create_summary_table(
                visible_predictions
            )
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_data = (
            pd.DataFrame(
                visible_predictions
            )
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "Download Predictions CSV",
            data=csv_data,
            file_name=(
                "predictions_"
                f"{prediction_date.date()}"
                ".csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        # ====================================================
        # DETAILED MATCH CARDS
        # ====================================================

        st.markdown(
            "### Match Details"
        )

        for match in (
            visible_predictions
        ):

            display_prediction_card(
                match
            )

    else:

        st.info(
            "There are no matches from "
            "the selected league on this date."
        )


# ============================================================
# FAILURES
# ============================================================

failures = (
    st.session_state[
        "failures"
    ]
)

if failures:

    with st.expander(
        f"⚠️ {len(failures)} fixture(s) "
        "could not be modeled"
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
# EXPLANATION
# ============================================================

with st.expander(
    "ℹ️ How this works"
):

    st.write(
        """
        The predictions are pre-match estimates.

        Before running predictions, the app checks
        Football-Data for newly completed matches.
        Any new results, shots on target and corners
        found there are added to the model data used
        for that prediction run.

        The Live Score Center is separate. It checks
        current match scores automatically but does
        not change a pre-match prediction while a game
        is being played.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictions are statistical estimates. "
    "Shots-on-target and corner models remain "
    "experimental and should be evaluated through "
    "historical and forward testing."
)
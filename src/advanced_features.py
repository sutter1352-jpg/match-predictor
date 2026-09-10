from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR


MAX_HISTORY = 10
VENUE_HISTORY = 5
DECAY = 0.75


def points_for_result(result, is_home):
    if result == "D":
        return 1

    if is_home:
        return 3 if result == "H" else 0

    return 3 if result == "A" else 0


def summarize(history, window):
    """
    Return pre-match averages using only games
    that happened BEFORE the current game.
    """

    games = list(history)[-window:]

    if len(games) == 0:
        return {
            "ppg": np.nan,
            "gfpg": np.nan,
            "gapg": np.nan,
            "gdpg": np.nan,
            "games": 0,
        }

    points = sum(x["points"] for x in games)
    goals_for = sum(x["goals_for"] for x in games)
    goals_against = sum(x["goals_against"] for x in games)

    n = len(games)

    return {
        "ppg": points / n,
        "gfpg": goals_for / n,
        "gapg": goals_against / n,
        "gdpg": (goals_for - goals_against) / n,
        "games": n,
    }


def exponential_summary(history):
    """
    More recent matches receive more weight.

    Newest match weight = 1.0
    Previous match = 0.75
    Previous = 0.75^2
    etc.
    """

    games = list(history)[-MAX_HISTORY:]

    if len(games) == 0:
        return {
            "ewm_ppg": np.nan,
            "ewm_gfpg": np.nan,
            "ewm_gapg": np.nan,
        }

    n = len(games)

    weights = DECAY ** np.arange(
        n - 1,
        -1,
        -1
    )

    weights = weights / weights.sum()

    points = np.array([
        x["points"]
        for x in games
    ])

    goals_for = np.array([
        x["goals_for"]
        for x in games
    ])

    goals_against = np.array([
        x["goals_against"]
        for x in games
    ])

    return {
        "ewm_ppg": np.sum(points * weights),
        "ewm_gfpg": np.sum(goals_for * weights),
        "ewm_gapg": np.sum(goals_against * weights),
    }


def rest_days(last_date, current_date):

    if last_date is None:
        return np.nan

    days = (
        current_date - last_date
    ).days

    # After ~2 weeks, additional rest
    # probably should not scale linearly.
    return min(days, 14)


def add_advanced_features(df):

    df = (
        df
        .sort_values(
            ["date", "match_id"]
        )
        .copy()
    )

    overall_history = defaultdict(
        lambda: deque(
            maxlen=MAX_HISTORY
        )
    )

    home_history = defaultdict(
        lambda: deque(
            maxlen=VENUE_HISTORY
        )
    )

    away_history = defaultdict(
        lambda: deque(
            maxlen=VENUE_HISTORY
        )
    )

    last_match_date = {}

    feature_rows = []

    for row in df.itertuples(index=False):

        home_key = (
            row.league_code,
            row.home_team
        )

        away_key = (
            row.league_code,
            row.away_team
        )

        # ----------------------------
        # Clear stale recent history
        # if team has been outside the
        # league for a long period.
        # ----------------------------

        for key in [home_key, away_key]:

            previous_date = (
                last_match_date.get(key)
            )

            if previous_date is not None:

                gap = (
                    row.date
                    - previous_date
                ).days

                if gap > 180:
                    overall_history[
                        key
                    ].clear()

                    home_history[
                        key
                    ].clear()

                    away_history[
                        key
                    ].clear()

        # ============================
        # OVERALL FORM
        # ============================

        h5 = summarize(
            overall_history[home_key],
            5
        )

        a5 = summarize(
            overall_history[away_key],
            5
        )

        h10 = summarize(
            overall_history[home_key],
            10
        )

        a10 = summarize(
            overall_history[away_key],
            10
        )

        # ============================
        # VENUE-SPECIFIC FORM
        # ============================

        home_at_home = summarize(
            home_history[home_key],
            5
        )

        away_away = summarize(
            away_history[away_key],
            5
        )

        # ============================
        # EXPONENTIAL FORM
        # ============================

        home_ewm = (
            exponential_summary(
                overall_history[
                    home_key
                ]
            )
        )

        away_ewm = (
            exponential_summary(
                overall_history[
                    away_key
                ]
            )
        )

        # ============================
        # REST
        # ============================

        home_rest = rest_days(
            last_match_date.get(
                home_key
            ),
            row.date
        )

        away_rest = rest_days(
            last_match_date.get(
                away_key
            ),
            row.date
        )

        # ============================
        # SAVE PRE-MATCH FEATURES
        # ============================

        features = {

            # Overall last 5
            "home_ppg_last5":
                h5["ppg"],

            "away_ppg_last5":
                a5["ppg"],

            "home_gfpg_last5":
                h5["gfpg"],

            "away_gfpg_last5":
                a5["gfpg"],

            "home_gapg_last5":
                h5["gapg"],

            "away_gapg_last5":
                a5["gapg"],

            "home_gdpg_last5":
                h5["gdpg"],

            "away_gdpg_last5":
                a5["gdpg"],


            # Overall last 10
            "home_ppg_last10":
                h10["ppg"],

            "away_ppg_last10":
                a10["ppg"],

            "home_gfpg_last10":
                h10["gfpg"],

            "away_gfpg_last10":
                a10["gfpg"],

            "home_gapg_last10":
                h10["gapg"],

            "away_gapg_last10":
                a10["gapg"],

            "home_gdpg_last10":
                h10["gdpg"],

            "away_gdpg_last10":
                a10["gdpg"],


            # Home / away form
            "home_home_ppg_last5":
                home_at_home["ppg"],

            "home_home_gfpg_last5":
                home_at_home["gfpg"],

            "home_home_gapg_last5":
                home_at_home["gapg"],

            "away_away_ppg_last5":
                away_away["ppg"],

            "away_away_gfpg_last5":
                away_away["gfpg"],

            "away_away_gapg_last5":
                away_away["gapg"],


            # Exponential form
            "home_ewm_ppg":
                home_ewm["ewm_ppg"],

            "away_ewm_ppg":
                away_ewm["ewm_ppg"],

            "home_ewm_gfpg":
                home_ewm["ewm_gfpg"],

            "away_ewm_gfpg":
                away_ewm["ewm_gfpg"],

            "home_ewm_gapg":
                home_ewm["ewm_gapg"],

            "away_ewm_gapg":
                away_ewm["ewm_gapg"],


            # Rest
            "home_rest_days":
                home_rest,

            "away_rest_days":
                away_rest,


            # Number of games known
            "home_games_available":
                h10["games"],

            "away_games_available":
                a10["games"],
        }

        feature_rows.append(
            features
        )

        # ============================
        # NOW update histories
        # AFTER storing features.
        # ============================

        home_points = (
            points_for_result(
                row.result,
                True
            )
        )

        away_points = (
            points_for_result(
                row.result,
                False
            )
        )

        home_game = {
            "points":
                home_points,

            "goals_for":
                row.home_goals,

            "goals_against":
                row.away_goals,
        }

        away_game = {
            "points":
                away_points,

            "goals_for":
                row.away_goals,

            "goals_against":
                row.home_goals,
        }

        overall_history[
            home_key
        ].append(home_game)

        overall_history[
            away_key
        ].append(away_game)

        home_history[
            home_key
        ].append(home_game)

        away_history[
            away_key
        ].append(away_game)

        last_match_date[
            home_key
        ] = row.date

        last_match_date[
            away_key
        ] = row.date

    feature_df = pd.DataFrame(
        feature_rows,
        index=df.index
    )

    df = pd.concat(
        [
            df,
            feature_df
        ],
        axis=1
    )

    return df


def run():

    input_path = (
        PROCESSED_DIR
        / "matches_with_elo.csv"
    )

    df = pd.read_csv(
        input_path,
        parse_dates=["date"]
    )

    print(
        f"Loaded {len(df):,} matches"
    )

    df = add_advanced_features(
        df
    )

    output_path = (
        PROCESSED_DIR
        / "matches_advanced_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved -> {output_path}"
    )

    print(
        "\nNew feature examples:\n"
    )

    columns = [
        "date",
        "home_team",
        "away_team",
        "home_ppg_last5",
        "away_ppg_last5",
        "home_ppg_last10",
        "away_ppg_last10",
        "home_home_ppg_last5",
        "away_away_ppg_last5",
        "home_rest_days",
        "away_rest_days",
    ]

    print(
        df[
            columns
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
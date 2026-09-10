import math
import difflib

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, LEAGUES

from .match_predictor import (
    load_matches,
    available_teams,
    resolve_team,
    predict_match,
    choose_league,
)


# ============================================================
# SETTINGS
# ============================================================

OVERALL_GAMES = 20
VENUE_GAMES = 12

HALF_LIFE_DAYS = 180
LEAGUE_HALF_LIFE_DAYS = 365

OVERALL_SHRINK = 6
VENUE_SHRINK = 5

# Pull predictions slightly toward league averages.
FINAL_REGRESSION = 0.20


# ============================================================
# COLUMN ALIASES
#
# The script automatically searches for the column names
# used in your dataset.
# ============================================================

COLUMN_ALIASES = {

    "home_sot": [
        "home_shots_on_target",
        "home_sot",
        "hst",
        "HST",
    ],

    "away_sot": [
        "away_shots_on_target",
        "away_sot",
        "ast",
        "AST",
    ],

    "home_corners": [
        "home_corners",
        "hc",
        "HC",
    ],

    "away_corners": [
        "away_corners",
        "ac",
        "AC",
    ],
}


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    aliases
):

    # Exact match first
    for alias in aliases:

        if alias in df.columns:
            return alias

    # Case-insensitive
    lookup = {
        column.lower(): column
        for column in df.columns
    }

    for alias in aliases:

        if alias.lower() in lookup:

            return lookup[
                alias.lower()
            ]

    return None


# ============================================================
# RESOLVE STAT COLUMNS
# ============================================================

def resolve_stat_columns(
    df
):

    columns = {}

    for name, aliases in (
        COLUMN_ALIASES.items()
    ):

        column = find_column(
            df,
            aliases
        )

        columns[
            name
        ] = column

    missing = [
        name
        for name, column
        in columns.items()
        if column is None
    ]

    if missing:

        print(
            "\nERROR: Could not find all "
            "required statistics columns."
        )

        print(
            "\nMissing:"
        )

        for name in missing:

            print(
                f"  {name}"
            )

        print(
            "\nColumns currently available:"
        )

        print(
            list(df.columns)
        )

        raise ValueError(
            "Missing shots-on-target "
            "or corner columns."
        )

    return columns


# ============================================================
# WEIGHTED MEAN
# ============================================================

def weighted_mean(
    values,
    dates,
    prediction_date,
    half_life
):

    if len(values) == 0:

        return np.nan

    values = np.asarray(
        values,
        dtype=float
    )

    dates = pd.Series(
        pd.to_datetime(
            dates
        )
    ).reset_index(
        drop=True
    )

    prediction_date = pd.Timestamp(
        prediction_date
    )

    age_days = (
        prediction_date
        - dates
    ).dt.days.to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(values)
        &
        np.isfinite(age_days)
    )

    values = values[
        valid
    ]

    age_days = age_days[
        valid
    ]

    if len(values) == 0:

        return np.nan

    age_days = np.maximum(
        age_days,
        0
    )

    weights = np.power(
        0.5,
        age_days
        / half_life
    )

    weight_sum = float(
        np.sum(
            weights
        )
    )

    if (
        not np.isfinite(
            weight_sum
        )
        or weight_sum <= 0
    ):

        return float(
            np.mean(
                values
            )
        )

    return float(
        np.average(
            values,
            weights=weights
        )
    )


# ============================================================
# SHRINKAGE
# ============================================================

def shrink(
    value,
    games,
    baseline,
    strength
):

    if (
        games <= 0
        or not np.isfinite(
            value
        )
    ):

        return float(
            baseline
        )

    return float(
        (
            games * value
            +
            strength * baseline
        )
        /
        (
            games
            +
            strength
        )
    )


# ============================================================
# LEAGUE METRIC AVERAGES
#
# Example:
#
# home SOT average
# away SOT average
# overall SOT average
# ============================================================

def league_metric_averages(
    matches,
    league_code,
    prediction_date,
    home_column,
    away_column
):

    league = matches[
        (
            matches[
                "league_code"
            ]
            == league_code
        )
        &
        (
            matches[
                "date"
            ]
            < prediction_date
        )
    ].copy()

    cutoff = (
        prediction_date
        - pd.Timedelta(
            days=730
        )
    )

    recent = league[
        league["date"]
        >= cutoff
    ]

    if len(recent) >= 100:

        league = recent

    home_values = pd.to_numeric(
        league[
            home_column
        ],
        errors="coerce"
    )

    away_values = pd.to_numeric(
        league[
            away_column
        ],
        errors="coerce"
    )

    home_average = weighted_mean(
        home_values,
        league["date"],
        prediction_date,
        LEAGUE_HALF_LIFE_DAYS
    )

    away_average = weighted_mean(
        away_values,
        league["date"],
        prediction_date,
        LEAGUE_HALF_LIFE_DAYS
    )

    if not np.isfinite(
        home_average
    ):

        home_average = float(
            home_values.mean()
        )

    if not np.isfinite(
        away_average
    ):

        away_average = float(
            away_values.mean()
        )

    overall_average = (
        home_average
        +
        away_average
    ) / 2

    return (
        float(
            home_average
        ),
        float(
            away_average
        ),
        float(
            overall_average
        ),
    )


# ============================================================
# BUILD TEAM METRIC HISTORY
#
# metric_for     = team's own count
# metric_against = opponent's count
# ============================================================

def team_metric_history(
    matches,
    team,
    league_code,
    prediction_date,
    home_column,
    away_column
):

    league = matches[
        (
            matches[
                "league_code"
            ]
            == league_code
        )
        &
        (
            matches[
                "date"
            ]
            < prediction_date
        )
    ]

    # --------------------------------------------------------
    # Team at home
    # --------------------------------------------------------

    home = league[
        league[
            "home_team"
        ]
        == team
    ][
        [
            "date",
            home_column,
            away_column,
        ]
    ].copy()

    home = home.rename(
        columns={
            home_column:
                "metric_for",

            away_column:
                "metric_against",
        }
    )

    home[
        "venue"
    ] = "home"

    # --------------------------------------------------------
    # Team away
    # --------------------------------------------------------

    away = league[
        league[
            "away_team"
        ]
        == team
    ][
        [
            "date",
            home_column,
            away_column,
        ]
    ].copy()

    away = away.rename(
        columns={
            away_column:
                "metric_for",

            home_column:
                "metric_against",
        }
    )

    away[
        "venue"
    ] = "away"

    history = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True
    )

    history[
        "metric_for"
    ] = pd.to_numeric(
        history[
            "metric_for"
        ],
        errors="coerce"
    )

    history[
        "metric_against"
    ] = pd.to_numeric(
        history[
            "metric_against"
        ],
        errors="coerce"
    )

    history = history.dropna(
        subset=[
            "metric_for",
            "metric_against",
        ]
    )

    return (
        history
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TEAM METRIC STRENGTH
# ============================================================

def team_metric_stats(
    matches,
    team,
    league_code,
    venue,
    prediction_date,
    home_column,
    away_column,
    home_league_average,
    away_league_average,
    overall_league_average
):

    history = team_metric_history(
        matches,
        team,
        league_code,
        prediction_date,
        home_column,
        away_column
    )

    if venue == "home":

        venue_for_baseline = (
            home_league_average
        )

        venue_against_baseline = (
            away_league_average
        )

    else:

        venue_for_baseline = (
            away_league_average
        )

        venue_against_baseline = (
            home_league_average
        )

    if len(history) == 0:

        return {
            "overall_for":
                overall_league_average,

            "overall_against":
                overall_league_average,

            "venue_for":
                venue_for_baseline,

            "venue_against":
                venue_against_baseline,
        }

    # ========================================================
    # OVERALL
    # ========================================================

    overall = history.tail(
        OVERALL_GAMES
    )

    overall_for = shrink(
        weighted_mean(
            overall[
                "metric_for"
            ],
            overall[
                "date"
            ],
            prediction_date,
            HALF_LIFE_DAYS
        ),
        len(overall),
        overall_league_average,
        OVERALL_SHRINK
    )

    overall_against = shrink(
        weighted_mean(
            overall[
                "metric_against"
            ],
            overall[
                "date"
            ],
            prediction_date,
            HALF_LIFE_DAYS
        ),
        len(overall),
        overall_league_average,
        OVERALL_SHRINK
    )

    # ========================================================
    # VENUE
    # ========================================================

    venue_history = history[
        history[
            "venue"
        ]
        == venue
    ].tail(
        VENUE_GAMES
    )

    venue_for = shrink(
        weighted_mean(
            venue_history[
                "metric_for"
            ],
            venue_history[
                "date"
            ],
            prediction_date,
            HALF_LIFE_DAYS
        ),
        len(
            venue_history
        ),
        venue_for_baseline,
        VENUE_SHRINK
    )

    venue_against = shrink(
        weighted_mean(
            venue_history[
                "metric_against"
            ],
            venue_history[
                "date"
            ],
            prediction_date,
            HALF_LIFE_DAYS
        ),
        len(
            venue_history
        ),
        venue_against_baseline,
        VENUE_SHRINK
    )

    return {
        "overall_for":
            overall_for,

        "overall_against":
            overall_against,

        "venue_for":
            venue_for,

        "venue_against":
            venue_against,
    }


# ============================================================
# PREDICT COUNT STAT
#
# Used for both:
#   shots on target
#   corners
# ============================================================

def predict_metric(
    matches,
    home_team,
    away_team,
    league_code,
    prediction_date,
    home_column,
    away_column
):

    (
        home_league_average,
        away_league_average,
        overall_league_average,
    ) = league_metric_averages(
        matches,
        league_code,
        prediction_date,
        home_column,
        away_column
    )

    home_stats = team_metric_stats(
        matches,
        home_team,
        league_code,
        "home",
        prediction_date,
        home_column,
        away_column,
        home_league_average,
        away_league_average,
        overall_league_average
    )

    away_stats = team_metric_stats(
        matches,
        away_team,
        league_code,
        "away",
        prediction_date,
        home_column,
        away_column,
        home_league_average,
        away_league_average,
        overall_league_average
    )

    # ========================================================
    # HOME TEAM
    # ========================================================

    home_attack = (
        (
            home_stats[
                "venue_for"
            ]
            /
            home_league_average
        )
        ** 0.60
        *
        (
            home_stats[
                "overall_for"
            ]
            /
            overall_league_average
        )
        ** 0.40
    )

    away_allows = (
        (
            away_stats[
                "venue_against"
            ]
            /
            home_league_average
        )
        ** 0.60
        *
        (
            away_stats[
                "overall_against"
            ]
            /
            overall_league_average
        )
        ** 0.40
    )

    expected_home = (
        home_league_average
        *
        home_attack
        *
        away_allows
    )

    # ========================================================
    # AWAY TEAM
    # ========================================================

    away_attack = (
        (
            away_stats[
                "venue_for"
            ]
            /
            away_league_average
        )
        ** 0.60
        *
        (
            away_stats[
                "overall_for"
            ]
            /
            overall_league_average
        )
        ** 0.40
    )

    home_allows = (
        (
            home_stats[
                "venue_against"
            ]
            /
            away_league_average
        )
        ** 0.60
        *
        (
            home_stats[
                "overall_against"
            ]
            /
            overall_league_average
        )
        ** 0.40
    )

    expected_away = (
        away_league_average
        *
        away_attack
        *
        home_allows
    )

    # ========================================================
    # REGRESSION
    # ========================================================

    expected_home = (
        (
            1
            - FINAL_REGRESSION
        )
        *
        expected_home
        +
        FINAL_REGRESSION
        *
        home_league_average
    )

    expected_away = (
        (
            1
            - FINAL_REGRESSION
        )
        *
        expected_away
        +
        FINAL_REGRESSION
        *
        away_league_average
    )

    # Count safety range
    expected_home = float(
        np.clip(
            expected_home,
            0.1,
            15.0
        )
    )

    expected_away = float(
        np.clip(
            expected_away,
            0.1,
            15.0
        )
    )

    return (
        expected_home,
        expected_away
    )


# ============================================================
# PREDICT INTEGER COUNT
#
# For display only.
# Expected values are more useful statistically,
# but this gives a simple single-number prediction.
# ============================================================

def predicted_count(
    expected
):

    return int(
        round(
            expected
        )
    )


# ============================================================
# ANALYZE MATCH
# ============================================================

def analyze_match(
    matches,
    stat_columns,
    home_team,
    away_team,
    league_code,
    prediction_date
):

    # ========================================================
    # RESULT
    # ========================================================

    result_prediction = (
        predict_match(
            matches,
            home_team,
            away_team,
            league_code,
            prediction_date
        )
    )

    probabilities = (
        result_prediction[
            "probabilities"
        ]
    )

    labels = [
        "H",
        "D",
        "A",
    ]

    best_result = labels[
        int(
            np.argmax(
                probabilities
            )
        )
    ]

    # ========================================================
    # SHOTS ON TARGET
    # ========================================================

    (
        home_sot,
        away_sot,
    ) = predict_metric(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date,
        stat_columns[
            "home_sot"
        ],
        stat_columns[
            "away_sot"
        ]
    )

    # ========================================================
    # CORNERS
    # ========================================================

    (
        home_corners,
        away_corners,
    ) = predict_metric(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date,
        stat_columns[
            "home_corners"
        ],
        stat_columns[
            "away_corners"
        ]
    )

    return {
        "probabilities":
            probabilities,

        "best_result":
            best_result,

        "home_sot":
            home_sot,

        "away_sot":
            away_sot,

        "home_corners":
            home_corners,

        "away_corners":
            away_corners,
    }


# ============================================================
# DISPLAY
# ============================================================

def display_prediction(
    home_team,
    away_team,
    prediction
):

    probabilities = (
        prediction[
            "probabilities"
        ]
    )

    best_result = (
        prediction[
            "best_result"
        ]
    )

    if best_result == "H":

        result_name = (
            f"{home_team} WIN"
        )

    elif best_result == "A":

        result_name = (
            f"{away_team} WIN"
        )

    else:

        result_name = "DRAW"

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{home_team} vs {away_team}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # RESULT
    # ========================================================

    print(
        "\nRESULT"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team:<30}"
        f"{probabilities[0]:>10.1%}"
    )

    print(
        f"{'DRAW':<30}"
        f"{probabilities[1]:>10.1%}"
    )

    print(
        f"{away_team:<30}"
        f"{probabilities[2]:>10.1%}"
    )

    print(
        f"\nPrediction: "
        f"{result_name}"
    )

    # ========================================================
    # SHOTS ON TARGET
    # ========================================================

    home_sot = (
        prediction[
            "home_sot"
        ]
    )

    away_sot = (
        prediction[
            "away_sot"
        ]
    )

    print(
        "\nSHOTS ON TARGET"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team:<30}"
        f"{home_sot:>7.2f}"
        f"   (~{predicted_count(home_sot)})"
    )

    print(
        f"{away_team:<30}"
        f"{away_sot:>7.2f}"
        f"   (~{predicted_count(away_sot)})"
    )

    print(
        f"{'TOTAL':<30}"
        f"{home_sot + away_sot:>7.2f}"
        f"   (~{predicted_count(home_sot + away_sot)})"
    )

    # ========================================================
    # CORNERS
    # ========================================================

    home_corners = (
        prediction[
            "home_corners"
        ]
    )

    away_corners = (
        prediction[
            "away_corners"
        ]
    )

    print(
        "\nCORNERS"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team:<30}"
        f"{home_corners:>7.2f}"
        f"   (~{predicted_count(home_corners)})"
    )

    print(
        f"{away_team:<30}"
        f"{away_corners:>7.2f}"
        f"   (~{predicted_count(away_corners)})"
    )

    print(
        f"{'TOTAL':<30}"
        f"{home_corners + away_corners:>7.2f}"
        f"   (~{predicted_count(home_corners + away_corners)})"
    )

    print(
        "\n"
        + "-" * 72
    )

    print(
        "Expected values are estimates, "
        "not guaranteed exact counts."
    )


# ============================================================
# MAIN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MATCH RESULT + STATS PREDICTOR"
    )

    print(
        "=" * 72
    )

    matches = load_matches()

    stat_columns = (
        resolve_stat_columns(
            matches
        )
    )

    print(
        f"\nMatches loaded: "
        f"{len(matches):,}"
    )

    print(
        "\nStatistics columns detected:"
    )

    print(
        f"Shots on target: "
        f"{stat_columns['home_sot']} / "
        f"{stat_columns['away_sot']}"
    )

    print(
        f"Corners: "
        f"{stat_columns['home_corners']} / "
        f"{stat_columns['away_corners']}"
    )

    while True:

        league_code = (
            choose_league()
        )

        if league_code is None:

            print(
                "Invalid league."
            )

            continue

        teams = available_teams(
            matches,
            league_code
        )

        print(
            f"\n{LEAGUES[league_code]}"
        )

        print(
            f"Current teams found: "
            f"{len(teams)}"
        )

        # ----------------------------------------------------
        # Teams
        # ----------------------------------------------------

        home_team = resolve_team(
            input(
                "\nHome team: "
            ).strip(),
            teams
        )

        if home_team is None:

            print(
                "Home team not found."
            )

            continue

        away_team = resolve_team(
            input(
                "Away team: "
            ).strip(),
            teams
        )

        if away_team is None:

            print(
                "Away team not found."
            )

            continue

        if home_team == away_team:

            print(
                "Teams cannot be the same."
            )

            continue

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        value = input(
            "Match date YYYY-MM-DD "
            "(Enter for today): "
        ).strip()

        if value == "":

            prediction_date = (
                pd.Timestamp.today()
                .normalize()
            )

        else:

            try:

                prediction_date = (
                    pd.Timestamp(
                        value
                    ).normalize()
                )

            except ValueError:

                print(
                    "Invalid date."
                )

                continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = analyze_match(
            matches,
            stat_columns,
            home_team,
            away_team,
            league_code,
            prediction_date
        )

        display_prediction(
            home_team,
            away_team,
            prediction
        )

        again = input(
            "\nPredict another match? "
            "(y/n): "
        ).strip().lower()

        if again not in {
            "y",
            "yes"
        }:

            break

    print(
        "\nPredictor closed."
    )


if __name__ == "__main__":
    run()
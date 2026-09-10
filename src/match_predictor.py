import math
import difflib
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, LEAGUES


# ============================================================
# SETTINGS
# ============================================================

MAX_GOALS = 8

FORM_HALF_LIFE_DAYS = 180
LEAGUE_HALF_LIFE_DAYS = 365

OVERALL_GAMES = 20
VENUE_GAMES = 12
RECENT_GAMES = 5

OVERALL_SHRINK = 6
VENUE_SHRINK = 5
RECENT_SHRINK = 4

STARTING_ELO = 1500
ELO_K = 20
HOME_ELO_ADVANTAGE = 65
ELO_XG_DENOMINATOR = 2200

FORM_STRENGTH = 0.04
FORM_FACTOR_MIN = 0.92
FORM_FACTOR_MAX = 1.08

LEAGUE_REGRESSION = 0.20

# Learned from the historical calibration test.
TEMPERATURE = 1.08

ACTIVE_TEAM_WINDOW_DAYS = 120


# ============================================================
# DATA
# ============================================================

def load_matches():
    path = PROCESSED_DIR / "matches_clean.csv"

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    if "league_code" not in df.columns:

        if "league" not in df.columns:
            raise ValueError(
                "matches_clean.csv must contain "
                "'league_code' or 'league'."
            )

        reverse_leagues = {
            league_name: code
            for code, league_name
            in LEAGUES.items()
        }

        df["league_code"] = (
            df["league"]
            .map(reverse_leagues)
        )

    required = [
        "date",
        "league_code",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["home_goals"] = pd.to_numeric(
        df["home_goals"],
        errors="coerce"
    )

    df["away_goals"] = pd.to_numeric(
        df["away_goals"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required
    ).copy()

    return (
        df.sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# HELPERS
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
        pd.to_datetime(dates)
    ).reset_index(drop=True)

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

    values = values[valid]
    age_days = age_days[valid]

    if len(values) == 0:
        return np.nan

    age_days = np.maximum(
        age_days,
        0.0
    )

    weights = np.power(
        0.5,
        age_days / half_life
    )

    if (
        not np.isfinite(weights).all()
        or float(np.sum(weights)) <= 0
    ):
        return float(
            np.mean(values)
        )

    return float(
        np.average(
            values,
            weights=weights
        )
    )


def shrink(
    value,
    sample_size,
    baseline,
    strength
):
    if (
        sample_size <= 0
        or not np.isfinite(value)
    ):
        return float(
            baseline
        )

    return float(
        (
            sample_size * value
            + strength * baseline
        )
        /
        (
            sample_size
            + strength
        )
    )


# ============================================================
# LEAGUE AVERAGES
# ============================================================

def league_averages(
    matches,
    league_code,
    prediction_date
):
    league = matches[
        (
            matches["league_code"]
            == league_code
        )
        &
        (
            matches["date"]
            < prediction_date
        )
    ].copy()

    recent_cutoff = (
        prediction_date
        - pd.Timedelta(days=730)
    )

    recent = league[
        league["date"]
        >= recent_cutoff
    ].copy()

    if len(recent) >= 100:
        league = recent

    home_avg = weighted_mean(
        league["home_goals"],
        league["date"],
        prediction_date,
        LEAGUE_HALF_LIFE_DAYS
    )

    away_avg = weighted_mean(
        league["away_goals"],
        league["date"],
        prediction_date,
        LEAGUE_HALF_LIFE_DAYS
    )

    if not np.isfinite(home_avg):
        home_avg = 1.45

    if not np.isfinite(away_avg):
        away_avg = 1.15

    overall_avg = (
        home_avg
        + away_avg
    ) / 2

    return (
        float(home_avg),
        float(away_avg),
        float(overall_avg),
    )


# ============================================================
# TEAM HISTORY
# ============================================================

def team_history(
    matches,
    team,
    league_code,
    prediction_date
):
    league = matches[
        (
            matches["league_code"]
            == league_code
        )
        &
        (
            matches["date"]
            < prediction_date
        )
    ]

    home = league[
        league["home_team"] == team
    ][
        [
            "date",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_goals": "gf",
            "away_goals": "ga",
        }
    )

    home["venue"] = "home"

    away = league[
        league["away_team"] == team
    ][
        [
            "date",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_goals": "gf",
            "home_goals": "ga",
        }
    )

    away["venue"] = "away"

    history = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True
    )

    history = (
        history
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(history) == 0:
        return history

    history["points"] = np.select(
        [
            history["gf"]
            > history["ga"],

            history["gf"]
            == history["ga"],
        ],
        [
            3,
            1,
        ],
        default=0
    )

    return history


# ============================================================
# TEAM STATISTICS
# ============================================================

def team_stats(
    matches,
    team,
    league_code,
    venue,
    prediction_date,
    home_league_avg,
    away_league_avg,
    overall_league_avg
):
    history = team_history(
        matches,
        team,
        league_code,
        prediction_date
    )

    if venue == "home":

        venue_gf_baseline = (
            home_league_avg
        )

        venue_ga_baseline = (
            away_league_avg
        )

    else:

        venue_gf_baseline = (
            away_league_avg
        )

        venue_ga_baseline = (
            home_league_avg
        )

    if len(history) == 0:

        return {
            "overall_gf":
                overall_league_avg,

            "overall_ga":
                overall_league_avg,

            "venue_gf":
                venue_gf_baseline,

            "venue_ga":
                venue_ga_baseline,

            "recent_gf":
                overall_league_avg,

            "recent_ga":
                overall_league_avg,

            "recent_gd":
                0.0,

            "recent_ppg":
                1.3,
        }

    # --------------------------------------------------------
    # Overall last 20
    # --------------------------------------------------------

    overall = history.tail(
        OVERALL_GAMES
    )

    overall_gf = shrink(
        weighted_mean(
            overall["gf"],
            overall["date"],
            prediction_date,
            FORM_HALF_LIFE_DAYS
        ),
        len(overall),
        overall_league_avg,
        OVERALL_SHRINK
    )

    overall_ga = shrink(
        weighted_mean(
            overall["ga"],
            overall["date"],
            prediction_date,
            FORM_HALF_LIFE_DAYS
        ),
        len(overall),
        overall_league_avg,
        OVERALL_SHRINK
    )

    # --------------------------------------------------------
    # Venue-specific
    # --------------------------------------------------------

    venue_history = history[
        history["venue"]
        == venue
    ].tail(
        VENUE_GAMES
    )

    venue_gf = shrink(
        weighted_mean(
            venue_history["gf"],
            venue_history["date"],
            prediction_date,
            FORM_HALF_LIFE_DAYS
        ),
        len(venue_history),
        venue_gf_baseline,
        VENUE_SHRINK
    )

    venue_ga = shrink(
        weighted_mean(
            venue_history["ga"],
            venue_history["date"],
            prediction_date,
            FORM_HALF_LIFE_DAYS
        ),
        len(venue_history),
        venue_ga_baseline,
        VENUE_SHRINK
    )

    # --------------------------------------------------------
    # Last 5
    # --------------------------------------------------------

    recent = history.tail(
        RECENT_GAMES
    )

    recent_gf = shrink(
        weighted_mean(
            recent["gf"],
            recent["date"],
            prediction_date,
            90
        ),
        len(recent),
        overall_league_avg,
        RECENT_SHRINK
    )

    recent_ga = shrink(
        weighted_mean(
            recent["ga"],
            recent["date"],
            prediction_date,
            90
        ),
        len(recent),
        overall_league_avg,
        RECENT_SHRINK
    )

    recent_ppg = weighted_mean(
        recent["points"],
        recent["date"],
        prediction_date,
        90
    )

    if not np.isfinite(
        recent_ppg
    ):
        recent_ppg = 1.3

    return {
        "overall_gf":
            float(overall_gf),

        "overall_ga":
            float(overall_ga),

        "venue_gf":
            float(venue_gf),

        "venue_ga":
            float(venue_ga),

        "recent_gf":
            float(recent_gf),

        "recent_ga":
            float(recent_ga),

        "recent_gd":
            float(
                recent_gf
                - recent_ga
            ),

        "recent_ppg":
            float(recent_ppg),
    }


# ============================================================
# ELO
# ============================================================

def calculate_current_elo(
    matches,
    league_code,
    prediction_date
):
    league = matches[
        (
            matches["league_code"]
            == league_code
        )
        &
        (
            matches["date"]
            < prediction_date
        )
    ].sort_values(
        "date"
    )

    ratings = defaultdict(
        lambda: STARTING_ELO
    )

    for row in league.itertuples():

        home = row.home_team
        away = row.away_team

        home_rating = (
            ratings[home]
        )

        away_rating = (
            ratings[away]
        )

        adjusted_home = (
            home_rating
            + HOME_ELO_ADVANTAGE
        )

        expected_home = (
            1
            /
            (
                1
                + 10
                ** (
                    (
                        away_rating
                        - adjusted_home
                    )
                    / 400
                )
            )
        )

        if (
            row.home_goals
            > row.away_goals
        ):

            actual_home = 1.0

        elif (
            row.home_goals
            == row.away_goals
        ):

            actual_home = 0.5

        else:

            actual_home = 0.0

        goal_difference = abs(
            row.home_goals
            - row.away_goals
        )

        margin_multiplier = min(
            1
            + 0.12
            * max(
                goal_difference - 1,
                0
            ),
            1.45
        )

        change = (
            ELO_K
            * margin_multiplier
            * (
                actual_home
                - expected_home
            )
        )

        ratings[home] = (
            home_rating
            + change
        )

        ratings[away] = (
            away_rating
            - change
        )

    return ratings


# ============================================================
# EXPECTED GOALS
# ============================================================

def expected_goals(
    matches,
    home_team,
    away_team,
    league_code,
    prediction_date
):
    (
        home_league_avg,
        away_league_avg,
        overall_league_avg,
    ) = league_averages(
        matches,
        league_code,
        prediction_date
    )

    home_stats = team_stats(
        matches,
        home_team,
        league_code,
        "home",
        prediction_date,
        home_league_avg,
        away_league_avg,
        overall_league_avg
    )

    away_stats = team_stats(
        matches,
        away_team,
        league_code,
        "away",
        prediction_date,
        home_league_avg,
        away_league_avg,
        overall_league_avg
    )

    # --------------------------------------------------------
    # Attack strength
    # --------------------------------------------------------

    home_attack = (
        (
            home_stats["venue_gf"]
            / home_league_avg
        )
        ** 0.65
        *
        (
            home_stats["overall_gf"]
            / overall_league_avg
        )
        ** 0.35
    )

    away_attack = (
        (
            away_stats["venue_gf"]
            / away_league_avg
        )
        ** 0.65
        *
        (
            away_stats["overall_gf"]
            / overall_league_avg
        )
        ** 0.35
    )

    # --------------------------------------------------------
    # Defensive weakness
    # --------------------------------------------------------

    away_defense_weakness = (
        (
            away_stats["venue_ga"]
            / home_league_avg
        )
        ** 0.65
        *
        (
            away_stats["overall_ga"]
            / overall_league_avg
        )
        ** 0.35
    )

    home_defense_weakness = (
        (
            home_stats["venue_ga"]
            / away_league_avg
        )
        ** 0.65
        *
        (
            home_stats["overall_ga"]
            / overall_league_avg
        )
        ** 0.35
    )

    # --------------------------------------------------------
    # Base xG
    # --------------------------------------------------------

    home_xg = (
        home_league_avg
        * home_attack
        * away_defense_weakness
    )

    away_xg = (
        away_league_avg
        * away_attack
        * home_defense_weakness
    )

    # --------------------------------------------------------
    # Elo adjustment
    # --------------------------------------------------------

    ratings = calculate_current_elo(
        matches,
        league_code,
        prediction_date
    )

    home_elo = float(
        ratings[home_team]
    )

    away_elo = float(
        ratings[away_team]
    )

    elo_difference = (
        home_elo
        + HOME_ELO_ADVANTAGE
        - away_elo
    )

    home_elo_factor = float(
        np.clip(
            math.exp(
                elo_difference
                / ELO_XG_DENOMINATOR
            ),
            0.82,
            1.22
        )
    )

    away_elo_factor = float(
        np.clip(
            math.exp(
                -elo_difference
                / ELO_XG_DENOMINATOR
            ),
            0.82,
            1.22
        )
    )

    home_xg *= (
        home_elo_factor
    )

    away_xg *= (
        away_elo_factor
    )

    # --------------------------------------------------------
    # Recent form
    # --------------------------------------------------------

    recent_difference = (
        home_stats["recent_gd"]
        - away_stats["recent_gd"]
    )

    form_factor = float(
        np.clip(
            math.exp(
                FORM_STRENGTH
                * recent_difference
            ),
            FORM_FACTOR_MIN,
            FORM_FACTOR_MAX
        )
    )

    home_xg *= (
        form_factor
    )

    away_xg /= (
        form_factor
    )

    # --------------------------------------------------------
    # Regress extreme xG toward league average
    # --------------------------------------------------------

    home_xg = (
        (
            1
            - LEAGUE_REGRESSION
        )
        * home_xg
        +
        LEAGUE_REGRESSION
        * home_league_avg
    )

    away_xg = (
        (
            1
            - LEAGUE_REGRESSION
        )
        * away_xg
        +
        LEAGUE_REGRESSION
        * away_league_avg
    )

    home_xg = float(
        np.clip(
            home_xg,
            0.20,
            3.75
        )
    )

    away_xg = float(
        np.clip(
            away_xg,
            0.20,
            3.75
        )
    )

    details = {
        "home_elo":
            home_elo,

        "away_elo":
            away_elo,

        "home_ppg":
            float(
                home_stats[
                    "recent_ppg"
                ]
            ),

        "away_ppg":
            float(
                away_stats[
                    "recent_ppg"
                ]
            ),
    }

    return (
        home_xg,
        away_xg,
        details
    )


# ============================================================
# POISSON SCORE MODEL
# ============================================================

def poisson_probability(
    goals,
    expected
):
    expected = max(
        float(expected),
        0.001
    )

    return (
        math.exp(-expected)
        * expected ** goals
        / math.factorial(goals)
    )


def score_matrix(
    home_xg,
    away_xg
):
    matrix = np.zeros(
        (
            MAX_GOALS + 1,
            MAX_GOALS + 1,
        ),
        dtype=float
    )

    for home_goals in range(
        MAX_GOALS + 1
    ):

        home_probability = (
            poisson_probability(
                home_goals,
                home_xg
            )
        )

        for away_goals in range(
            MAX_GOALS + 1
        ):

            matrix[
                home_goals,
                away_goals
            ] = (
                home_probability
                * poisson_probability(
                    away_goals,
                    away_xg
                )
            )

    total = float(
        matrix.sum()
    )

    if total <= 0:
        raise ValueError(
            "Could not create score probability matrix."
        )

    return (
        matrix / total
    )


def result_probabilities(
    matrix
):
    home = np.tril(
        matrix,
        k=-1
    ).sum()

    draw = np.trace(
        matrix
    )

    away = np.triu(
        matrix,
        k=1
    ).sum()

    probabilities = np.array(
        [
            home,
            draw,
            away,
        ],
        dtype=float
    )

    return (
        probabilities
        / probabilities.sum()
    )


# ============================================================
# H/D/A TEMPERATURE CALIBRATION
# ============================================================

def calibrate_probabilities(
    probabilities,
    temperature=TEMPERATURE
):
    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    logits = np.log(
        probabilities
    )

    scaled_logits = (
        logits
        / temperature
    )

    scaled_logits -= (
        np.max(
            scaled_logits
        )
    )

    exp_logits = np.exp(
        scaled_logits
    )

    return (
        exp_logits
        / exp_logits.sum()
    )


# ============================================================
# SCORELINE HELPERS
# ============================================================

def top_scores(
    matrix,
    count=5
):
    scores = []

    for home_goals in range(
        matrix.shape[0]
    ):

        for away_goals in range(
            matrix.shape[1]
        ):

            scores.append(
                (
                    home_goals,
                    away_goals,
                    float(
                        matrix[
                            home_goals,
                            away_goals
                        ]
                    ),
                )
            )

    scores.sort(
        key=lambda item:
        item[2],
        reverse=True
    )

    return scores[:count]


def conditional_score(
    matrix,
    result
):
    best = None

    for home_goals in range(
        matrix.shape[0]
    ):

        for away_goals in range(
            matrix.shape[1]
        ):

            if result == "H":

                valid = (
                    home_goals
                    > away_goals
                )

            elif result == "D":

                valid = (
                    home_goals
                    == away_goals
                )

            else:

                valid = (
                    away_goals
                    > home_goals
                )

            if not valid:
                continue

            probability = float(
                matrix[
                    home_goals,
                    away_goals
                ]
            )

            if (
                best is None
                or probability > best[2]
            ):

                best = (
                    home_goals,
                    away_goals,
                    probability,
                )

    return best


# ============================================================
# TEAMS
# ============================================================

def available_teams(
    matches,
    league_code
):
    league = matches[
        matches["league_code"]
        == league_code
    ].copy()

    if len(league) == 0:
        return []

    latest_date = (
        league["date"].max()
    )

    cutoff = (
        latest_date
        - pd.Timedelta(
            days=
            ACTIVE_TEAM_WINDOW_DAYS
        )
    )

    recent = league[
        league["date"]
        >= cutoff
    ]

    teams = set(
        recent["home_team"]
    )

    teams.update(
        recent["away_team"]
    )

    return sorted(
        teams
    )


def resolve_team(
    typed_name,
    teams
):
    typed_name = (
        typed_name.strip()
    )

    lookup = {
        team.lower():
        team
        for team in teams
    }

    if (
        typed_name.lower()
        in lookup
    ):

        return lookup[
            typed_name.lower()
        ]

    partial = [
        team
        for team in teams
        if (
            typed_name.lower()
            in team.lower()
        )
    ]

    if len(partial) == 1:

        print(
            f"Using: "
            f"{partial[0]}"
        )

        return partial[0]

    fuzzy = (
        difflib.get_close_matches(
            typed_name,
            teams,
            n=5,
            cutoff=0.40
        )
    )

    if len(fuzzy) == 0:
        return None

    if len(fuzzy) == 1:

        print(
            f"Using: "
            f"{fuzzy[0]}"
        )

        return fuzzy[0]

    print(
        "\nDid you mean:"
    )

    for index, team in enumerate(
        fuzzy,
        start=1
    ):

        print(
            f"{index}. {team}"
        )

    choice = input(
        "Choose number: "
    ).strip()

    try:

        index = (
            int(choice)
            - 1
        )

        if (
            index < 0
            or index
            >= len(fuzzy)
        ):
            return None

        return fuzzy[index]

    except ValueError:
        return None


# ============================================================
# OPTIONAL MARKET COMPARISON
# ============================================================

def market_probabilities(
    odds
):
    implied = np.array(
        [
            1 / odds[0],
            1 / odds[1],
            1 / odds[2],
        ],
        dtype=float
    )

    return (
        implied
        / implied.sum()
    )


def ask_for_odds():
    print(
        "\nOPTIONAL CURRENT ODDS"
    )

    print(
        "Enter Home Draw Away decimal odds."
    )

    print(
        "Example: 1.80 3.60 4.50"
    )

    value = input(
        "Odds "
        "(press Enter to skip): "
    ).strip()

    if value == "":
        return None

    try:

        odds = [
            float(x)
            for x in value.split()
        ]

        if (
            len(odds) != 3
            or any(
                odd <= 1
                for odd in odds
            )
        ):
            raise ValueError

        return odds

    except ValueError:

        print(
            "Invalid odds. "
            "Market comparison skipped."
        )

        return None


# ============================================================
# PREDICT MATCH
# ============================================================

def predict_match(
    matches,
    home_team,
    away_team,
    league_code,
    prediction_date
):
    (
        home_xg,
        away_xg,
        details,
    ) = expected_goals(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date
    )

    matrix = score_matrix(
        home_xg,
        away_xg
    )

    raw_probabilities = (
        result_probabilities(
            matrix
        )
    )

    calibrated_probabilities = (
        calibrate_probabilities(
            raw_probabilities
        )
    )

    labels = np.array(
        [
            "H",
            "D",
            "A",
        ]
    )

    best_result = labels[
        int(
            np.argmax(
                calibrated_probabilities
            )
        )
    ]

    return {
        "home_xg":
            home_xg,

        "away_xg":
            away_xg,

        "details":
            details,

        "matrix":
            matrix,

        "raw_probabilities":
            raw_probabilities,

        "probabilities":
            calibrated_probabilities,

        "best_result":
            best_result,

        "scores":
            top_scores(
                matrix,
                count=5
            ),

        "conditional_score":
            conditional_score(
                matrix,
                best_result
            ),
    }


# ============================================================
# DISPLAY
# ============================================================

def display_prediction(
    home_team,
    away_team,
    prediction,
    odds=None
):
    probabilities = (
        prediction[
            "probabilities"
        ]
    )

    raw_probabilities = (
        prediction[
            "raw_probabilities"
        ]
    )

    home_xg = (
        prediction[
            "home_xg"
        ]
    )

    away_xg = (
        prediction[
            "away_xg"
        ]
    )

    best_result = (
        prediction[
            "best_result"
        ]
    )

    scores = (
        prediction[
            "scores"
        ]
    )

    conditional = (
        prediction[
            "conditional_score"
        ]
    )

    details = (
        prediction[
            "details"
        ]
    )

    names = [
        home_team,
        "DRAW",
        away_team,
    ]

    result_index = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{home_team} "
        f"vs "
        f"{away_team}"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Calibrated result probabilities
    # --------------------------------------------------------

    print(
        "\nCALIBRATED MATCH RESULT PROBABILITIES"
    )

    print(
        "-" * 48
    )

    for (
        name,
        probability,
    ) in zip(
        names,
        probabilities
    ):

        print(
            f"{name:<32}"
            f"{probability:>8.1%}"
        )

    # --------------------------------------------------------
    # Most probable result
    # --------------------------------------------------------

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

    best_index = (
        result_index[
            best_result
        ]
    )

    ordered = np.sort(
        probabilities
    )[::-1]

    gap = (
        ordered[0]
        - ordered[1]
    )

    if gap >= 0.20:

        confidence = "HIGH"

    elif gap >= 0.10:

        confidence = "MODERATE"

    else:

        confidence = (
            "LOW / CLOSE MATCH"
        )

    print(
        "\nMOST PROBABLE RESULT"
    )

    print(
        "-" * 48
    )

    print(
        result_name
    )

    print(
        "Calibrated probability: "
        f"{probabilities[best_index]:.1%}"
    )

    print(
        "Prediction confidence: "
        f"{confidence}"
    )

    # --------------------------------------------------------
    # xG
    # --------------------------------------------------------

    print(
        "\nEXPECTED GOALS"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team:<32}"
        f"{home_xg:>8.2f}"
    )

    print(
        f"{away_team:<32}"
        f"{away_xg:>8.2f}"
    )

    # --------------------------------------------------------
    # Most likely score
    # --------------------------------------------------------

    best_score = (
        scores[0]
    )

    print(
        "\nMOST LIKELY EXACT SCORE"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team} "
        f"{best_score[0]}"
        " - "
        f"{best_score[1]} "
        f"{away_team}"
    )

    print(
        "Exact-score probability: "
        f"{best_score[2]:.1%}"
    )

    # --------------------------------------------------------
    # Conditional score
    # --------------------------------------------------------

    print(
        "\nMOST LIKELY SCORE "
        "FOR THE PREDICTED RESULT"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team} "
        f"{conditional[0]}"
        " - "
        f"{conditional[1]} "
        f"{away_team}"
    )

    print(
        "Exact-score probability: "
        f"{conditional[2]:.1%}"
    )

    # --------------------------------------------------------
    # Top 5
    # --------------------------------------------------------

    print(
        "\nTOP 5 SCORELINES"
    )

    print(
        "-" * 48
    )

    for (
        home_score,
        away_score,
        probability,
    ) in scores:

        print(
            f"{home_score}-"
            f"{away_score:<8}"
            f"{probability:>8.1%}"
        )

    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    print(
        "\nCURRENT TEAM STRENGTH"
    )

    print(
        "-" * 48
    )

    print(
        f"{home_team} Elo: "
        f"{details['home_elo']:.0f}"
    )

    print(
        f"{away_team} Elo: "
        f"{details['away_elo']:.0f}"
    )

    print(
        f"{home_team} recent PPG: "
        f"{details['home_ppg']:.2f}"
    )

    print(
        f"{away_team} recent PPG: "
        f"{details['away_ppg']:.2f}"
    )

    # --------------------------------------------------------
    # Calibration details
    # --------------------------------------------------------

    print(
        "\nCALIBRATION"
    )

    print(
        "-" * 48
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE:.2f}"
    )

    print(
        "Raw -> Calibrated"
    )

    for (
        name,
        raw,
        calibrated,
    ) in zip(
        names,
        raw_probabilities,
        probabilities
    ):

        print(
            f"{name:<20}"
            f"{raw:>7.1%}"
            " -> "
            f"{calibrated:>7.1%}"
        )

    # --------------------------------------------------------
    # Optional market comparison
    # --------------------------------------------------------

    if odds is not None:

        market = (
            market_probabilities(
                odds
            )
        )

        differences = (
            probabilities
            - market
        )

        print(
            "\nCURRENT MARKET "
            "FAIR PROBABILITIES"
        )

        print(
            "-" * 48
        )

        for (
            name,
            probability,
        ) in zip(
            names,
            market
        ):

            print(
                f"{name:<32}"
                f"{probability:>8.1%}"
            )

        print(
            "\nCALIBRATED MODEL VS MARKET"
        )

        print(
            "-" * 48
        )

        for (
            name,
            difference,
        ) in zip(
            names,
            differences
        ):

            print(
                f"{name:<32}"
                f"{difference:>+8.1%}"
            )

        print(
            "\nThis difference alone is "
            "NOT a betting recommendation."
        )

    print(
        "\n"
        + "-" * 72
    )

    print(
        "Prediction only. "
        "Most probable does not mean guaranteed."
    )


# ============================================================
# LEAGUE MENU
# ============================================================

def choose_league():
    print(
        "\nSUPPORTED LEAGUES"
    )

    print(
        "-" * 42
    )

    codes = list(
        LEAGUES.keys()
    )

    for (
        index,
        code,
    ) in enumerate(
        codes,
        start=1
    ):

        print(
            f"{index}. "
            f"{LEAGUES[code]} "
            f"({code})"
        )

    value = input(
        "\nChoose league: "
    ).strip()

    if (
        value.upper()
        in LEAGUES
    ):

        return value.upper()

    try:

        index = (
            int(value)
            - 1
        )

        if (
            index < 0
            or index
            >= len(codes)
        ):

            return None

        return codes[index]

    except ValueError:

        return None


# ============================================================
# MAIN
# ============================================================

def run():
    print(
        "\n"
        + "=" * 72
    )

    print(
        "CALIBRATED DYNAMIC SOCCER MATCH PREDICTOR"
    )

    print(
        "=" * 72
    )

    matches = (
        load_matches()
    )

    prediction_date = (
        pd.Timestamp.today()
        .normalize()
    )

    print(
        f"\nMatches loaded: "
        f"{len(matches):,}"
    )

    print(
        f"Prediction date: "
        f"{prediction_date.date()}"
    )

    print(
        f"Probability calibration: "
        f"temperature "
        f"{TEMPERATURE:.2f}"
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

        if len(teams) == 0:

            print(
                "No active teams found."
            )

            continue

        print(
            f"\n"
            f"{LEAGUES[league_code]}"
        )

        print(
            f"Current teams found: "
            f"{len(teams)}"
        )

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

        if (
            home_team
            == away_team
        ):

            print(
                "Home and away team "
                "cannot be the same."
            )

            continue

        odds = (
            ask_for_odds()
        )

        prediction = (
            predict_match(
                matches,
                home_team,
                away_team,
                league_code,
                prediction_date
            )
        )

        display_prediction(
            home_team,
            away_team,
            prediction,
            odds
        )

        again = input(
            "\nPredict another match? "
            "(y/n): "
        ).strip().lower()

        if again not in [
            "y",
            "yes",
        ]:

            break

    print(
        "\nPredictor closed."
    )


if __name__ == "__main__":
    run()
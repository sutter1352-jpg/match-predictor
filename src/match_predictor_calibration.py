import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, LEAGUES


# ============================================================
# SETTINGS
# Keep these aligned with src/match_predictor.py
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

TEST_START = pd.Timestamp("2022-08-01")
TEST_END = pd.Timestamp("2026-08-01")


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

    dates = pd.to_datetime(
        pd.Series(
            list(dates)
        )
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

    weight_sum = float(
        np.sum(weights)
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= 0
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


# ============================================================
# SHRINK SMALL SAMPLES TOWARD LEAGUE AVERAGE
# ============================================================

def shrink(
    value,
    sample_size,
    baseline,
    strength
):

    if (
        sample_size <= 0
        or
        not np.isfinite(value)
    ):
        return float(
            baseline
        )

    return float(
        (
            sample_size * value
            +
            strength * baseline
        )
        /
        (
            sample_size
            +
            strength
        )
    )


# ============================================================
# POISSON
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
        math.exp(
            -expected
        )
        *
        expected ** goals
        /
        math.factorial(
            goals
        )
    )


# ============================================================
# HOME / DRAW / AWAY PROBABILITIES
# ============================================================

def result_probabilities(
    home_xg,
    away_xg
):

    matrix = np.zeros(
        (
            MAX_GOALS + 1,
            MAX_GOALS + 1
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
                *
                poisson_probability(
                    away_goals,
                    away_xg
                )
            )

    total = matrix.sum()

    if total <= 0:

        return np.array([
            1 / 3,
            1 / 3,
            1 / 3,
        ])

    matrix /= total

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

    probabilities = np.array([
        home,
        draw,
        away,
    ])

    return (
        probabilities
        /
        probabilities.sum()
    )


# ============================================================
# LEAGUE WALK-FORWARD STATE
# ============================================================

class LeagueState:

    def __init__(self):

        self.ratings = defaultdict(
            lambda: STARTING_ELO
        )

        self.overall_history = defaultdict(
            lambda: deque(
                maxlen=OVERALL_GAMES
            )
        )

        self.home_history = defaultdict(
            lambda: deque(
                maxlen=VENUE_GAMES
            )
        )

        self.away_history = defaultdict(
            lambda: deque(
                maxlen=VENUE_GAMES
            )
        )

        self.league_history = deque()


    # ========================================================
    # LEAGUE HISTORY
    # ========================================================

    def prune_league_history(
        self,
        prediction_date
    ):

        cutoff = (
            pd.Timestamp(
                prediction_date
            )
            -
            pd.Timedelta(
                days=730
            )
        )

        while (
            self.league_history
            and
            self.league_history[0]["date"]
            < cutoff
        ):

            self.league_history.popleft()


    # ========================================================
    # LEAGUE AVERAGES
    # ========================================================

    def league_averages(
        self,
        prediction_date
    ):

        self.prune_league_history(
            prediction_date
        )

        if len(
            self.league_history
        ) == 0:

            return (
                1.45,
                1.15,
                1.30,
            )

        dates = [
            row["date"]
            for row
            in self.league_history
        ]

        home_goals = [
            row["home_goals"]
            for row
            in self.league_history
        ]

        away_goals = [
            row["away_goals"]
            for row
            in self.league_history
        ]

        home_average = weighted_mean(
            home_goals,
            dates,
            prediction_date,
            LEAGUE_HALF_LIFE_DAYS
        )

        away_average = weighted_mean(
            away_goals,
            dates,
            prediction_date,
            LEAGUE_HALF_LIFE_DAYS
        )

        if not np.isfinite(
            home_average
        ):
            home_average = 1.45

        if not np.isfinite(
            away_average
        ):
            away_average = 1.15

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


    # ========================================================
    # TEAM STATS
    # ========================================================

    def team_stats(
        self,
        team,
        venue,
        prediction_date,
        home_league_avg,
        away_league_avg,
        overall_league_avg,
    ):

        overall = list(
            self.overall_history[
                team
            ]
        )

        if venue == "home":

            venue_history = list(
                self.home_history[
                    team
                ]
            )

            venue_gf_baseline = (
                home_league_avg
            )

            venue_ga_baseline = (
                away_league_avg
            )

        else:

            venue_history = list(
                self.away_history[
                    team
                ]
            )

            venue_gf_baseline = (
                away_league_avg
            )

            venue_ga_baseline = (
                home_league_avg
            )

        if len(overall) == 0:

            return {
                "overall_gf":
                    overall_league_avg,

                "overall_ga":
                    overall_league_avg,

                "venue_gf":
                    venue_gf_baseline,

                "venue_ga":
                    venue_ga_baseline,

                "recent_gd":
                    0.0,

                "recent_ppg":
                    1.3,
            }

        # ====================================================
        # OVERALL FORM
        # ====================================================

        overall_dates = [
            row["date"]
            for row
            in overall
        ]

        overall_gf_raw = (
            weighted_mean(
                [
                    row["gf"]
                    for row
                    in overall
                ],
                overall_dates,
                prediction_date,
                FORM_HALF_LIFE_DAYS
            )
        )

        overall_ga_raw = (
            weighted_mean(
                [
                    row["ga"]
                    for row
                    in overall
                ],
                overall_dates,
                prediction_date,
                FORM_HALF_LIFE_DAYS
            )
        )

        overall_gf = shrink(
            overall_gf_raw,
            len(overall),
            overall_league_avg,
            OVERALL_SHRINK
        )

        overall_ga = shrink(
            overall_ga_raw,
            len(overall),
            overall_league_avg,
            OVERALL_SHRINK
        )

        # ====================================================
        # VENUE FORM
        # ====================================================

        if len(
            venue_history
        ) > 0:

            venue_dates = [
                row["date"]
                for row
                in venue_history
            ]

            venue_gf_raw = (
                weighted_mean(
                    [
                        row["gf"]
                        for row
                        in venue_history
                    ],
                    venue_dates,
                    prediction_date,
                    FORM_HALF_LIFE_DAYS
                )
            )

            venue_ga_raw = (
                weighted_mean(
                    [
                        row["ga"]
                        for row
                        in venue_history
                    ],
                    venue_dates,
                    prediction_date,
                    FORM_HALF_LIFE_DAYS
                )
            )

        else:

            venue_gf_raw = (
                venue_gf_baseline
            )

            venue_ga_raw = (
                venue_ga_baseline
            )

        venue_gf = shrink(
            venue_gf_raw,
            len(venue_history),
            venue_gf_baseline,
            VENUE_SHRINK
        )

        venue_ga = shrink(
            venue_ga_raw,
            len(venue_history),
            venue_ga_baseline,
            VENUE_SHRINK
        )

        # ====================================================
        # RECENT FORM
        # ====================================================

        recent = overall[
            -RECENT_GAMES:
        ]

        recent_dates = [
            row["date"]
            for row
            in recent
        ]

        recent_gf_raw = (
            weighted_mean(
                [
                    row["gf"]
                    for row
                    in recent
                ],
                recent_dates,
                prediction_date,
                90
            )
        )

        recent_ga_raw = (
            weighted_mean(
                [
                    row["ga"]
                    for row
                    in recent
                ],
                recent_dates,
                prediction_date,
                90
            )
        )

        recent_gf = shrink(
            recent_gf_raw,
            len(recent),
            overall_league_avg,
            RECENT_SHRINK
        )

        recent_ga = shrink(
            recent_ga_raw,
            len(recent),
            overall_league_avg,
            RECENT_SHRINK
        )

        recent_ppg = (
            weighted_mean(
                [
                    row["points"]
                    for row
                    in recent
                ],
                recent_dates,
                prediction_date,
                90
            )
        )

        if not np.isfinite(
            recent_ppg
        ):
            recent_ppg = 1.3

        return {
            "overall_gf":
                float(
                    overall_gf
                ),

            "overall_ga":
                float(
                    overall_ga
                ),

            "venue_gf":
                float(
                    venue_gf
                ),

            "venue_ga":
                float(
                    venue_ga
                ),

            "recent_gd":
                float(
                    recent_gf
                    -
                    recent_ga
                ),

            "recent_ppg":
                float(
                    recent_ppg
                ),
        }


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        home_team,
        away_team,
        prediction_date
    ):

        (
            home_league_avg,
            away_league_avg,
            overall_league_avg,
        ) = self.league_averages(
            prediction_date
        )

        home_stats = self.team_stats(
            home_team,
            "home",
            prediction_date,
            home_league_avg,
            away_league_avg,
            overall_league_avg,
        )

        away_stats = self.team_stats(
            away_team,
            "away",
            prediction_date,
            home_league_avg,
            away_league_avg,
            overall_league_avg,
        )

        # ====================================================
        # ATTACK
        # ====================================================

        home_attack = (
            (
                home_stats[
                    "venue_gf"
                ]
                /
                home_league_avg
            )
            ** 0.65
            *
            (
                home_stats[
                    "overall_gf"
                ]
                /
                overall_league_avg
            )
            ** 0.35
        )

        away_attack = (
            (
                away_stats[
                    "venue_gf"
                ]
                /
                away_league_avg
            )
            ** 0.65
            *
            (
                away_stats[
                    "overall_gf"
                ]
                /
                overall_league_avg
            )
            ** 0.35
        )

        # ====================================================
        # DEFENSIVE WEAKNESS
        # ====================================================

        away_defense = (
            (
                away_stats[
                    "venue_ga"
                ]
                /
                home_league_avg
            )
            ** 0.65
            *
            (
                away_stats[
                    "overall_ga"
                ]
                /
                overall_league_avg
            )
            ** 0.35
        )

        home_defense = (
            (
                home_stats[
                    "venue_ga"
                ]
                /
                away_league_avg
            )
            ** 0.65
            *
            (
                home_stats[
                    "overall_ga"
                ]
                /
                overall_league_avg
            )
            ** 0.35
        )

        # ====================================================
        # BASE XG
        # ====================================================

        home_xg = (
            home_league_avg
            *
            home_attack
            *
            away_defense
        )

        away_xg = (
            away_league_avg
            *
            away_attack
            *
            home_defense
        )

        # ====================================================
        # ELO
        # ====================================================

        home_elo = (
            self.ratings[
                home_team
            ]
        )

        away_elo = (
            self.ratings[
                away_team
            ]
        )

        elo_difference = (
            home_elo
            +
            HOME_ELO_ADVANTAGE
            -
            away_elo
        )

        home_elo_factor = (
            math.exp(
                elo_difference
                /
                ELO_XG_DENOMINATOR
            )
        )

        away_elo_factor = (
            math.exp(
                -elo_difference
                /
                ELO_XG_DENOMINATOR
            )
        )

        home_elo_factor = float(
            np.clip(
                home_elo_factor,
                0.82,
                1.22
            )
        )

        away_elo_factor = float(
            np.clip(
                away_elo_factor,
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

        # ====================================================
        # RECENT FORM
        # ====================================================

        recent_difference = (
            home_stats[
                "recent_gd"
            ]
            -
            away_stats[
                "recent_gd"
            ]
        )

        form_factor = (
            math.exp(
                FORM_STRENGTH
                *
                recent_difference
            )
        )

        form_factor = float(
            np.clip(
                form_factor,
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

        # ====================================================
        # REGRESS XG TOWARD LEAGUE AVERAGE
        # ====================================================

        home_xg = (
            (
                1
                -
                LEAGUE_REGRESSION
            )
            *
            home_xg
            +
            LEAGUE_REGRESSION
            *
            home_league_avg
        )

        away_xg = (
            (
                1
                -
                LEAGUE_REGRESSION
            )
            *
            away_xg
            +
            LEAGUE_REGRESSION
            *
            away_league_avg
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

        probabilities = (
            result_probabilities(
                home_xg,
                away_xg
            )
        )

        return (
            probabilities,
            home_xg,
            away_xg
        )


    # ========================================================
    # UPDATE STATE AFTER RESULT
    # ========================================================

    def update(
        self,
        date,
        home_team,
        away_team,
        home_goals,
        away_goals
    ):

        date = pd.Timestamp(
            date
        )

        # ====================================================
        # ELO UPDATE
        # ====================================================

        home_rating = (
            self.ratings[
                home_team
            ]
        )

        away_rating = (
            self.ratings[
                away_team
            ]
        )

        adjusted_home = (
            home_rating
            +
            HOME_ELO_ADVANTAGE
        )

        expected_home = (
            1.0
            /
            (
                1.0
                +
                10
                ** (
                    (
                        away_rating
                        -
                        adjusted_home
                    )
                    /
                    400.0
                )
            )
        )

        if home_goals > away_goals:

            actual_home = 1.0
            home_points = 3
            away_points = 0

        elif home_goals < away_goals:

            actual_home = 0.0
            home_points = 0
            away_points = 3

        else:

            actual_home = 0.5
            home_points = 1
            away_points = 1

        goal_difference = abs(
            home_goals
            -
            away_goals
        )

        margin_multiplier = min(
            1.0
            +
            0.12
            *
            max(
                goal_difference - 1,
                0
            ),
            1.45
        )

        change = (
            ELO_K
            *
            margin_multiplier
            *
            (
                actual_home
                -
                expected_home
            )
        )

        self.ratings[
            home_team
        ] = (
            home_rating
            +
            change
        )

        self.ratings[
            away_team
        ] = (
            away_rating
            -
            change
        )

        # ====================================================
        # TEAM HISTORY
        # ====================================================

        home_record = {
            "date":
                date,

            "gf":
                float(
                    home_goals
                ),

            "ga":
                float(
                    away_goals
                ),

            "points":
                float(
                    home_points
                ),
        }

        away_record = {
            "date":
                date,

            "gf":
                float(
                    away_goals
                ),

            "ga":
                float(
                    home_goals
                ),

            "points":
                float(
                    away_points
                ),
        }

        self.overall_history[
            home_team
        ].append(
            home_record
        )

        self.home_history[
            home_team
        ].append(
            home_record
        )

        self.overall_history[
            away_team
        ].append(
            away_record
        )

        self.away_history[
            away_team
        ].append(
            away_record
        )

        # ====================================================
        # LEAGUE HISTORY
        # ====================================================

        self.league_history.append({
            "date":
                date,

            "home_goals":
                float(
                    home_goals
                ),

            "away_goals":
                float(
                    away_goals
                ),
        })


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    path = (
        PROCESSED_DIR
        /
        "matches_clean.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=[
            "date"
        ]
    )

    if (
        "league_code"
        not in df.columns
    ):

        if (
            "league"
            not in df.columns
        ):

            raise ValueError(
                "matches_clean.csv needs "
                "'league_code' or 'league'."
            )

        reverse = {
            name: code
            for code, name
            in LEAGUES.items()
        }

        df[
            "league_code"
        ] = (
            df["league"]
            .map(reverse)
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
            "Missing columns: "
            +
            ", ".join(
                missing
            )
        )

    df[
        "home_goals"
    ] = pd.to_numeric(
        df["home_goals"],
        errors="coerce"
    )

    df[
        "away_goals"
    ] = pd.to_numeric(
        df["away_goals"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required
    ).copy()

    if (
        "result"
        not in df.columns
    ):

        df["result"] = np.where(
            df["home_goals"]
            >
            df["away_goals"],
            "H",
            np.where(
                df["home_goals"]
                <
                df["away_goals"],
                "A",
                "D"
            )
        )

    return (
        df
        .sort_values(
            [
                "date",
                "league_code",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def backtest(
    df
):

    states = {
        code: LeagueState()
        for code
        in LEAGUES.keys()
    }

    print(
        "\nBuilding model state "
        "from matches before 2022/23..."
    )

    historical = df[
        df["date"]
        < TEST_START
    ]

    for row in historical.itertuples():

        if (
            row.league_code
            not in states
        ):
            continue

        states[
            row.league_code
        ].update(
            row.date,
            row.home_team,
            row.away_team,
            int(
                row.home_goals
            ),
            int(
                row.away_goals
            ),
        )

    test = df[
        (
            df["date"]
            >= TEST_START
        )
        &
        (
            df["date"]
            < TEST_END
        )
    ].copy()

    predictions = []

    dates = sorted(
        test[
            "date"
        ].unique()
    )

    print(
        f"Testing "
        f"{len(test):,} matches..."
    )

    # ========================================================
    # WALK THROUGH EACH DATE
    # ========================================================

    for (
        day_number,
        date,
    ) in enumerate(
        dates,
        start=1
    ):

        date = pd.Timestamp(
            date
        )

        day = test[
            test["date"]
            == date
        ]

        # ====================================================
        # PREDICT FIRST
        # ====================================================

        for row in day.itertuples():

            if (
                row.league_code
                not in states
            ):
                continue

            (
                probabilities,
                home_xg,
                away_xg,
            ) = states[
                row.league_code
            ].predict(
                row.home_team,
                row.away_team,
                date
            )

            labels = np.array([
                "H",
                "D",
                "A",
            ])

            predicted_result = (
                labels[
                    int(
                        np.argmax(
                            probabilities
                        )
                    )
                ]
            )

            predictions.append({
                "date":
                    date,

                "league_code":
                    row.league_code,

                "league":
                    LEAGUES.get(
                        row.league_code,
                        row.league_code
                    ),

                "home_team":
                    row.home_team,

                "away_team":
                    row.away_team,

                "actual_result":
                    row.result,

                "predicted_result":
                    predicted_result,

                "correct":
                    (
                        predicted_result
                        ==
                        row.result
                    ),

                "confidence":
                    float(
                        np.max(
                            probabilities
                        )
                    ),

                "home_probability":
                    float(
                        probabilities[0]
                    ),

                "draw_probability":
                    float(
                        probabilities[1]
                    ),

                "away_probability":
                    float(
                        probabilities[2]
                    ),

                "home_xg":
                    home_xg,

                "away_xg":
                    away_xg,
            })

        # ====================================================
        # UPDATE ONLY AFTER ALL MATCHES THAT DAY ARE PREDICTED
        # ====================================================

        for row in day.itertuples():

            if (
                row.league_code
                not in states
            ):
                continue

            states[
                row.league_code
            ].update(
                date,
                row.home_team,
                row.away_team,
                int(
                    row.home_goals
                ),
                int(
                    row.away_goals
                ),
            )

        if (
            day_number
            % 100
            == 0
        ):

            print(
                f"Processed "
                f"{day_number}/"
                f"{len(dates)} "
                f"match days..."
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    predictions
):

    probabilities = predictions[
        [
            "home_probability",
            "draw_probability",
            "away_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    label_to_index = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    actual_index = np.array([
        label_to_index[
            result
        ]
        for result
        in predictions[
            "actual_result"
        ].to_numpy()
    ])

    actual_probabilities = (
        probabilities[
            np.arange(
                len(probabilities)
            ),
            actual_index
        ]
    )

    log_loss = -float(
        np.mean(
            np.log(
                np.clip(
                    actual_probabilities,
                    1e-15,
                    1.0
                )
            )
        )
    )

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(
            len(probabilities)
        ),
        actual_index
    ] = 1.0

    brier = float(
        np.mean(
            np.sum(
                (
                    probabilities
                    -
                    one_hot
                )
                ** 2,
                axis=1
            )
        )
    )

    accuracy = float(
        predictions[
            "correct"
        ].mean()
    )

    return (
        accuracy,
        log_loss,
        brier
    )


# ============================================================
# MOST-PROBABLE RESULT CALIBRATION
# ============================================================

def make_confidence_calibration(
    predictions
):

    bins = [
        0.33,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    labels = [
        "33-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    df = predictions.copy()

    df[
        "confidence_bin"
    ] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    summary = (
        df
        .dropna(
            subset=[
                "confidence_bin"
            ]
        )
        .groupby(
            "confidence_bin",
            observed=True
        )
        .agg(
            predictions=(
                "correct",
                "size"
            ),
            average_prediction=(
                "confidence",
                "mean"
            ),
            actual_accuracy=(
                "correct",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "calibration_gap"
    ] = (
        summary[
            "actual_accuracy"
        ]
        -
        summary[
            "average_prediction"
        ]
    )

    summary[
        "status"
    ] = np.where(
        summary[
            "calibration_gap"
        ].abs()
        <= 0.03,
        "GOOD",
        np.where(
            summary[
                "calibration_gap"
            ]
            < 0,
            "OVERCONFIDENT",
            "UNDERCONFIDENT"
        )
    )

    return summary


# ============================================================
# ALL H/D/A CALIBRATION
# ============================================================

def make_outcome_calibration(
    predictions
):

    rows = []

    columns = [
        (
            "H",
            "home_probability"
        ),
        (
            "D",
            "draw_probability"
        ),
        (
            "A",
            "away_probability"
        ),
    ]

    for row in predictions.itertuples():

        for (
            outcome,
            column,
        ) in columns:

            rows.append({
                "probability":
                    float(
                        getattr(
                            row,
                            column
                        )
                    ),

                "occurred":
                    float(
                        row.actual_result
                        ==
                        outcome
                    ),
            })

    df = pd.DataFrame(
        rows
    )

    bins = np.linspace(
        0.0,
        1.0,
        11
    )

    labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    df[
        "probability_bin"
    ] = pd.cut(
        df["probability"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True
    )

    summary = (
        df
        .dropna(
            subset=[
                "probability_bin"
            ]
        )
        .groupby(
            "probability_bin",
            observed=True
        )
        .agg(
            predictions=(
                "occurred",
                "size"
            ),
            average_prediction=(
                "probability",
                "mean"
            ),
            actual_frequency=(
                "occurred",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "calibration_gap"
    ] = (
        summary[
            "actual_frequency"
        ]
        -
        summary[
            "average_prediction"
        ]
    )

    return summary


# ============================================================
# DISPLAY TABLE
# ============================================================

def print_percent_table(
    df,
    actual_column
):

    display = (
        df.copy()
    )

    display[
        "average_prediction"
    ] *= 100

    display[
        actual_column
    ] *= 100

    display[
        "calibration_gap"
    ] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "average_prediction":
                    lambda x:
                    f"{x:.1f}%",

                actual_column:
                    lambda x:
                    f"{x:.1f}%",

                "calibration_gap":
                    lambda x:
                    f"{x:+.1f}%",
            }
        )
    )


# ============================================================
# RUN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DYNAMIC MATCH PREDICTOR CALIBRATION"
    )

    print(
        "=" * 78
    )

    print(
        "Walk-forward test: "
        "every game is predicted before "
        "its result is added."
    )

    df = load_data()

    print(
        f"\nDataset matches: "
        f"{len(df):,}"
    )

    predictions = (
        backtest(
            df
        )
    )

    (
        accuracy,
        log_loss,
        brier,
    ) = metrics(
        predictions
    )

    # ========================================================
    # OVERALL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "OVERALL BACKTEST"
    )

    print(
        "=" * 78
    )

    print(
        f"Matches:     "
        f"{len(predictions):,}"
    )

    print(
        f"Accuracy:    "
        f"{accuracy:.2%}"
    )

    print(
        f"Log Loss:    "
        f"{log_loss:.4f}"
    )

    print(
        f"Brier Score: "
        f"{brier:.4f}"
    )

    # ========================================================
    # MOST-PROBABLE RESULT CALIBRATION
    # ========================================================

    confidence = (
        make_confidence_calibration(
            predictions
        )
    )

    print(
        "\nMOST-PROBABLE RESULT CALIBRATION"
    )

    print(
        "=" * 78
    )

    print_percent_table(
        confidence,
        "actual_accuracy"
    )

    # ========================================================
    # ALL OUTCOME CALIBRATION
    # ========================================================

    outcome = (
        make_outcome_calibration(
            predictions
        )
    )

    print(
        "\nALL H/D/A PROBABILITY CALIBRATION"
    )

    print(
        "=" * 78
    )

    print_percent_table(
        outcome,
        "actual_frequency"
    )

    # ========================================================
    # SPECIAL 70-80% CHECK
    # ========================================================

    seventy = predictions[
        (
            predictions[
                "confidence"
            ]
            >= 0.70
        )
        &
        (
            predictions[
                "confidence"
            ]
            < 0.80
        )
    ]

    print(
        "\n"
        + "=" * 78
    )

    print(
        "70-80% PREDICTIONS — SPECIAL CHECK"
    )

    print(
        "=" * 78
    )

    if len(
        seventy
    ) == 0:

        print(
            "No predictions in "
            "the 70-80% range."
        )

    else:

        average_predicted = float(
            seventy[
                "confidence"
            ].mean()
        )

        actual_accuracy = float(
            seventy[
                "correct"
            ].mean()
        )

        gap = (
            actual_accuracy
            -
            average_predicted
        )

        print(
            f"Predictions:       "
            f"{len(seventy):,}"
        )

        print(
            f"Average predicted: "
            f"{average_predicted:.1%}"
        )

        print(
            f"Actually correct:  "
            f"{actual_accuracy:.1%}"
        )

        print(
            f"Calibration gap:   "
            f"{gap:+.1%}"
        )

        if abs(
            gap
        ) <= 0.03:

            print(
                "Result: "
                "REASONABLY CALIBRATED"
            )

        elif gap < 0:

            print(
                "Result: "
                "OVERCONFIDENT"
            )

        else:

            print(
                "Result: "
                "UNDERCONFIDENT"
            )

    # ========================================================
    # SAVE
    # ========================================================

    predictions.to_csv(
        PROCESSED_DIR
        /
        "match_predictor_backtest_predictions.csv",
        index=False
    )

    confidence.to_csv(
        PROCESSED_DIR
        /
        "match_predictor_calibration.csv",
        index=False
    )

    outcome.to_csv(
        PROCESSED_DIR
        /
        "match_predictor_outcome_calibration.csv",
        index=False
    )

    print(
        "\nCalibration files saved "
        "in data/processed/"
    )


if __name__ == "__main__":
    run()
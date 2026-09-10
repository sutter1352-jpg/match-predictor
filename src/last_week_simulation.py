import numpy as np
import pandas as pd
import requests

from sklearn.metrics import log_loss

from .config import (
    RAW_DIR,
    PROCESSED_DIR,
    LEAGUES,
)

from .build_dataset import build_dataset
from .elo import run as run_elo
from .advanced_features import run as run_advanced_features

from .ridge_market_movement import (
    prepare_data,
    FEATURES,
    build_model,
    logits_to_probabilities,
    fair_probabilities,
)


# ============================================================
# SIMULATION DATES
# ============================================================

START_DATE = pd.Timestamp("2026-09-03")
END_DATE = pd.Timestamp("2026-09-10")

CURRENT_SEASON = "2627"

# These are the five leagues we currently support.
LEAGUE_CODES = [
    "E0",
    "SP1",
    "I1",
    "D1",
    "F1",
]


# ============================================================
# RIDGE REGULARIZATION OPTIONS
# ============================================================

ALPHAS = [
    0.01,
    0.1,
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


# ============================================================
# DOWNLOAD CURRENT 2026/27 DATA
# ============================================================

def download_current_season():

    print("\nDOWNLOADING 2026/27 DATA")
    print("=" * 60)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for league_code in LEAGUE_CODES:

        url = (
            "https://www.football-data.co.uk/"
            f"mmz4281/{CURRENT_SEASON}/"
            f"{league_code}.csv"
        )

        destination = (
            RAW_DIR
            / f"{CURRENT_SEASON}_{league_code}.csv"
        )

        print(
            f"Downloading "
            f"{LEAGUES[league_code]}..."
        )

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent":
                    "soccer-prediction-research/1.0"
            }
        )

        response.raise_for_status()

        destination.write_bytes(
            response.content
        )

        print(
            f"Saved -> {destination.name}"
        )


# ============================================================
# PROBABILITY HELPERS
# ============================================================

def get_market_probs(df):

    return df[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()


def get_closing_probs(df):

    return fair_probabilities(

        df[
            "avg_close_home_odds"
        ].to_numpy(),

        df[
            "avg_close_draw_odds"
        ].to_numpy(),

        df[
            "avg_close_away_odds"
        ].to_numpy(),
    )


def calculate_log_loss(
    actual,
    probabilities
):

    # Convert:
    #
    # H, D, A
    #
    # to sklearn:
    #
    # A, D, H

    sklearn_probs = np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])

    return log_loss(
        actual,
        sklearn_probs,
        labels=[
            "A",
            "D",
            "H"
        ]
    )


def accuracy(
    actual,
    probabilities
):

    labels = np.array([
        "H",
        "D",
        "A"
    ])

    predictions = labels[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    return (
        predictions
        == actual
    ).mean()


# ============================================================
# SELECT ALPHA USING 2025/26
#
# IMPORTANT:
# Last week's games are NOT used to select alpha.
# ============================================================

def select_alpha(df):

    training = df[
        df["date"]
        < pd.Timestamp(
            "2025-08-01"
        )
    ].copy()

    validation = df[
        (
            df["date"]
            >= pd.Timestamp(
                "2025-08-01"
            )
        )
        &
        (
            df["date"]
            < pd.Timestamp(
                "2026-08-01"
            )
        )
    ].copy()

    closing = get_closing_probs(
        validation
    )

    best_alpha = None
    best_mae = float("inf")

    print(
        "\nSELECTING ALPHA USING 2025/26"
    )

    print(
        "=" * 60
    )

    for alpha in ALPHAS:

        home_model = build_model(
            alpha
        )

        away_model = build_model(
            alpha
        )

        home_model.fit(
            training[FEATURES],
            training[
                "target_home_move"
            ]
        )

        away_model.fit(
            training[FEATURES],
            training[
                "target_away_move"
            ]
        )

        home_move = (
            home_model.predict(
                validation[FEATURES]
            )
        )

        away_move = (
            away_model.predict(
                validation[FEATURES]
            )
        )

        prediction = (
            logits_to_probabilities(

                validation[
                    "market_home_logit"
                ].to_numpy()
                + home_move,

                validation[
                    "market_away_logit"
                ].to_numpy()
                + away_move
            )
        )

        mae = np.mean(
            np.abs(
                prediction
                - closing
            )
        )

        print(
            f"Alpha {alpha:<7} "
            f"Closing MAE: {mae:.5f}"
        )

        if mae < best_mae:

            best_mae = mae
            best_alpha = alpha

    print(
        "\nSelected alpha:",
        best_alpha
    )

    print(
        "Validation MAE:",
        f"{best_mae:.5f}"
    )

    return best_alpha


# ============================================================
# PREDICT ONE DAY
#
# The model is retrained using only matches that happened
# BEFORE that date.
# ============================================================

def predict_day(
    full_data,
    matches,
    prediction_date,
    alpha
):

    train = full_data[
        full_data["date"]
        < prediction_date
    ].copy()

    home_model = build_model(
        alpha
    )

    away_model = build_model(
        alpha
    )

    home_model.fit(
        train[FEATURES],
        train[
            "target_home_move"
        ]
    )

    away_model.fit(
        train[FEATURES],
        train[
            "target_away_move"
        ]
    )

    home_move = (
        home_model.predict(
            matches[FEATURES]
        )
    )

    away_move = (
        away_model.predict(
            matches[FEATURES]
        )
    )

    probabilities = (
        logits_to_probabilities(

            matches[
                "market_home_logit"
            ].to_numpy()
            + home_move,

            matches[
                "market_away_logit"
            ].to_numpy()
            + away_move
        )
    )

    return probabilities


# ============================================================
# RUN LAST-WEEK REPLAY
# ============================================================

def run():

    # --------------------------------------------------------
    # STEP 1
    # Refresh current season.
    # --------------------------------------------------------

    download_current_season()

    # --------------------------------------------------------
    # STEP 2
    # Rebuild the complete dataset including 2026/27.
    # --------------------------------------------------------

    print(
        "\nREBUILDING DATASET"
    )

    print(
        "=" * 60
    )

    build_dataset()

    print(
        "\nREBUILDING ELO"
    )

    print(
        "=" * 60
    )

    run_elo()

    print(
        "\nREBUILDING ADVANCED FEATURES"
    )

    print(
        "=" * 60
    )

    run_advanced_features()

    # --------------------------------------------------------
    # STEP 3
    # Load updated dataset.
    # --------------------------------------------------------

    df = pd.read_csv(

        PROCESSED_DIR
        / "matches_advanced_features.csv",

        parse_dates=["date"]
    )

    df, _, _ = prepare_data(
        df
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # STEP 4
    # Select alpha WITHOUT using last week.
    # --------------------------------------------------------

    alpha = select_alpha(
        df
    )

    # --------------------------------------------------------
    # STEP 5
    # Isolate last week's games.
    # --------------------------------------------------------

    target = df[
        (
            df["date"]
            >= START_DATE
        )
        &
        (
            df["date"]
            < END_DATE
        )
    ].copy()

    target = (
        target
        .sort_values(
            [
                "date",
                "league",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\nLAST-WEEK SIMULATION"
    )

    print(
        "=" * 60
    )

    print(
        "Date range:",
        START_DATE.date(),
        "to",
        (
            END_DATE
            - pd.Timedelta(days=1)
        ).date()
    )

    print(
        "Matches available:",
        len(target)
    )

    if len(target) == 0:

        print(
            "\nNo matches with both "
            "market and closing odds "
            "were found."
        )

        return

    # --------------------------------------------------------
    # STEP 6
    # Replay each day.
    # --------------------------------------------------------

    results = []

    dates = sorted(
        target["date"].unique()
    )

    for date in dates:

        date = pd.Timestamp(
            date
        )

        day = target[
            target["date"]
            == date
        ].copy()

        print(
            f"\nPredicting {date.date()}"
        )

        print(
            f"Matches: {len(day)}"
        )

        predicted = predict_day(
            df,
            day,
            date,
            alpha
        )

        market = get_market_probs(
            day
        )

        closing = get_closing_probs(
            day
        )

        for i, row in enumerate(
            day.itertuples()
        ):

            results.append({

                "date":
                    row.date,

                "league":
                    row.league,

                "home_team":
                    row.home_team,

                "away_team":
                    row.away_team,

                "result":
                    row.result,

                # Current/non-closing market
                "market_home":
                    market[i, 0],

                "market_draw":
                    market[i, 1],

                "market_away":
                    market[i, 2],

                # Ridge-adjusted probability
                "model_home":
                    predicted[i, 0],

                "model_draw":
                    predicted[i, 1],

                "model_away":
                    predicted[i, 2],

                # Actual closing market
                "close_home":
                    closing[i, 0],

                "close_draw":
                    closing[i, 1],

                "close_away":
                    closing[i, 2],
            })

    results = pd.DataFrame(
        results
    )

    # ========================================================
    # MATCH RESULT PERFORMANCE
    # ========================================================

    actual = (
        results["result"]
        .to_numpy()
    )

    market_probs = results[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()

    model_probs = results[
        [
            "model_home",
            "model_draw",
            "model_away",
        ]
    ].to_numpy()

    closing_probs = results[
        [
            "close_home",
            "close_draw",
            "close_away",
        ]
    ].to_numpy()

    print(
        "\n\nMATCH RESULT PERFORMANCE"
    )

    print(
        "=" * 60
    )

    print(
        "\nNON-CLOSING MARKET"
    )

    print(
        f"Accuracy: "
        f"{accuracy(actual, market_probs):.3f}"
    )

    print(
        f"Log Loss: "
        f"{calculate_log_loss(actual, market_probs):.4f}"
    )

    print(
        "\nRIDGE ADJUSTED MODEL"
    )

    print(
        f"Accuracy: "
        f"{accuracy(actual, model_probs):.3f}"
    )

    print(
        f"Log Loss: "
        f"{calculate_log_loss(actual, model_probs):.4f}"
    )

    print(
        "\nACTUAL CLOSING MARKET"
    )

    print(
        f"Accuracy: "
        f"{accuracy(actual, closing_probs):.3f}"
    )

    print(
        f"Log Loss: "
        f"{calculate_log_loss(actual, closing_probs):.4f}"
    )

    # ========================================================
    # CLOSING-LINE PREDICTION
    # ========================================================

    market_mae = np.mean(
        np.abs(
            market_probs
            - closing_probs
        )
    )

    model_mae = np.mean(
        np.abs(
            model_probs
            - closing_probs
        )
    )

    print(
        "\n\nCLOSING-LINE PERFORMANCE"
    )

    print(
        "=" * 60
    )

    print(
        f"Market -> Closing MAE: "
        f"{market_mae:.4f}"
    )

    print(
        f"Model -> Closing MAE:  "
        f"{model_mae:.4f}"
    )

    # ========================================================
    # CREATE STRONGEST SIGNAL PER MATCH
    # ========================================================

    predicted_moves = (
        model_probs
        - market_probs
    )

    actual_moves = (
        closing_probs
        - market_probs
    )

    strongest = np.argmax(
        np.abs(predicted_moves),
        axis=1
    )

    outcome_names = np.array([
        "HOME",
        "DRAW",
        "AWAY",
    ])

    signals = []

    for i in range(
        len(results)
    ):

        j = strongest[i]

        predicted_move = (
            predicted_moves[i, j]
        )

        actual_move = (
            actual_moves[i, j]
        )

        signals.append({

            "date":
                results.iloc[i]["date"],

            "league":
                results.iloc[i]["league"],

            "home_team":
                results.iloc[i]["home_team"],

            "away_team":
                results.iloc[i]["away_team"],

            "outcome":
                outcome_names[j],

            "predicted_move":
                predicted_move,

            "actual_move":
                actual_move,

            "directional_clv":
                (
                    np.sign(
                        predicted_move
                    )
                    * actual_move
                ),

            "direction_correct":
                (
                    np.sign(
                        predicted_move
                    )
                    ==
                    np.sign(
                        actual_move
                    )
                ),
        })

    signals = pd.DataFrame(
        signals
    )

    signals[
        "signal_strength"
    ] = (
        signals[
            "predicted_move"
        ]
        .abs()
    )

    # ========================================================
    # SIGNAL TEST
    # ========================================================

    print(
        "\n\nSIGNAL PERFORMANCE"
    )

    print(
        "=" * 60
    )

    for threshold in [
        0.0025,
        0.0050,
    ]:

        subset = signals[
            signals[
                "signal_strength"
            ]
            >= threshold
        ]

        print(
            f"\nSignal >= "
            f"{threshold:.2%}"
        )

        print(
            f"Signals: "
            f"{len(subset)}"
        )

        if len(subset) > 0:

            print(
                f"Direction accuracy: "
                f"{subset['direction_correct'].mean():.2%}"
            )

            print(
                f"Average CLV: "
                f"{subset['directional_clv'].mean():.3%}"
            )

    # ========================================================
    # PRINT STRONG SIGNALS
    # ========================================================

    strong = signals[
        signals[
            "signal_strength"
        ]
        >= 0.005
    ].copy()

    strong = strong.sort_values(
        "signal_strength",
        ascending=False
    )

    print(
        "\n\nSTRONG SIGNALS >= 0.50%"
    )

    print(
        "=" * 90
    )

    if len(strong) == 0:

        print(
            "No >=0.50% signals "
            "this week."
        )

    else:

        display = strong[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "outcome",
                "predicted_move",
                "actual_move",
                "direction_correct",
            ]
        ].copy()

        display[
            "predicted_move"
        ] *= 100

        display[
            "actual_move"
        ] *= 100

        print(
            display.to_string(
                index=False,
                formatters={

                    "predicted_move":
                        lambda x:
                        f"{x:+.2f}%",

                    "actual_move":
                        lambda x:
                        f"{x:+.2f}%",
                }
            )
        )

    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    results.to_csv(

        PROCESSED_DIR
        / "last_week_predictions.csv",

        index=False
    )

    signals.to_csv(

        PROCESSED_DIR
        / "last_week_signals.csv",

        index=False
    )

    print(
        "\n\nSaved:"
    )

    print(
        "data/processed/"
        "last_week_predictions.csv"
    )

    print(
        "data/processed/"
        "last_week_signals.csv"
    )


if __name__ == "__main__":
    run()
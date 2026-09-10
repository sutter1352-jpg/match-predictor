import numpy as np
import pandas as pd

from .config import PROCESSED_DIR


THRESHOLDS = [
    0.0000,
    0.0025,
    0.0050,
    0.0100,
    0.0150,
    0.0200,
    0.0300,
]


def run():

    path = (
        PROCESSED_DIR
        / "ridge_market_predictions.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    # ========================================================
    # CALCULATE PREDICTED MOVEMENT
    # ========================================================

    df["pred_move_home"] = (
        df["ridge_home"]
        - df["market_home"]
    )

    df["pred_move_draw"] = (
        df["ridge_draw"]
        - df["market_draw"]
    )

    df["pred_move_away"] = (
        df["ridge_away"]
        - df["market_away"]
    )

    # ========================================================
    # ACTUAL CLOSING MOVEMENT
    # ========================================================

    df["actual_move_home"] = (
        df["close_home"]
        - df["market_home"]
    )

    df["actual_move_draw"] = (
        df["close_draw"]
        - df["market_draw"]
    )

    df["actual_move_away"] = (
        df["close_away"]
        - df["market_away"]
    )

    # ========================================================
    # CHOOSE ONE SIGNAL PER MATCH
    #
    # Pick the outcome with the largest ABSOLUTE predicted move.
    # ========================================================

    signals = []

    outcomes = [
        "home",
        "draw",
        "away"
    ]

    for row in df.itertuples():

        predicted_moves = {

            "home":
                row.pred_move_home,

            "draw":
                row.pred_move_draw,

            "away":
                row.pred_move_away,
        }

        strongest_outcome = max(

            outcomes,

            key=lambda x:
                abs(
                    predicted_moves[x]
                )
        )

        predicted_move = (
            predicted_moves[
                strongest_outcome
            ]
        )

        actual_move = getattr(
            row,
            f"actual_move_{strongest_outcome}"
        )

        # Positive means actual closing market moved
        # in the direction we predicted.

        directional_clv = (
            np.sign(predicted_move)
            * actual_move
        )

        correct_direction = (
            np.sign(predicted_move)
            ==
            np.sign(actual_move)
        )

        signals.append({

            "date":
                row.date,

            "league":
                row.league,

            "home_team":
                row.home_team,

            "away_team":
                row.away_team,

            "outcome":
                strongest_outcome,

            "predicted_move":
                predicted_move,

            "predicted_move_abs":
                abs(predicted_move),

            "actual_move":
                actual_move,

            "directional_clv":
                directional_clv,

            "correct_direction":
                correct_direction,
        })

    signals = pd.DataFrame(
        signals
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print(
        "\nSIGNAL STRENGTH ANALYSIS"
    )

    print(
        "=============================================================="
    )

    print(
        "Threshold | Signals | Direction | Avg CLV | Median CLV"
    )

    print(
        "--------------------------------------------------------------"
    )

    results = []

    for threshold in THRESHOLDS:

        subset = signals[
            signals[
                "predicted_move_abs"
            ]
            >= threshold
        ].copy()

        if len(subset) == 0:
            continue

        direction_accuracy = (
            subset[
                "correct_direction"
            ]
            .mean()
        )

        average_clv = (
            subset[
                "directional_clv"
            ]
            .mean()
        )

        median_clv = (
            subset[
                "directional_clv"
            ]
            .median()
        )

        print(
            f"{threshold:>8.2%} | "
            f"{len(subset):>7} | "
            f"{direction_accuracy:>8.2%} | "
            f"{average_clv:>7.2%} | "
            f"{median_clv:>10.2%}"
        )

        results.append({

            "threshold":
                threshold,

            "signals":
                len(subset),

            "direction_accuracy":
                direction_accuracy,

            "average_clv":
                average_clv,

            "median_clv":
                median_clv,
        })

    # ========================================================
    # LEAGUE ANALYSIS
    # ========================================================

    print(
        "\n\nBY LEAGUE — SIGNALS >= 0.5%"
    )

    print(
        "=============================================================="
    )

    league_signals = signals[
        signals[
            "predicted_move_abs"
        ]
        >= 0.005
    ].copy()

    league_summary = (
        league_signals
        .groupby("league")
        .agg(

            signals=(
                "correct_direction",
                "size"
            ),

            direction_accuracy=(
                "correct_direction",
                "mean"
            ),

            average_clv=(
                "directional_clv",
                "mean"
            ),

            average_predicted_move=(
                "predicted_move_abs",
                "mean"
            ),
        )
        .sort_values(
            "average_clv",
            ascending=False
        )
    )

    print(
        league_summary.to_string(
            formatters={

                "direction_accuracy":
                    lambda x:
                    f"{x:.2%}",

                "average_clv":
                    lambda x:
                    f"{x:.2%}",

                "average_predicted_move":
                    lambda x:
                    f"{x:.2%}",
            }
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    signals.to_csv(

        PROCESSED_DIR
        / "ridge_signals.csv",

        index=False
    )

    pd.DataFrame(
        results
    ).to_csv(

        PROCESSED_DIR
        / "signal_threshold_results.csv",

        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "data/processed/ridge_signals.csv"
    )

    print(
        "data/processed/signal_threshold_results.csv"
    )


if __name__ == "__main__":
    run()
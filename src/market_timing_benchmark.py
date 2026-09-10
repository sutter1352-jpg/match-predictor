import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, accuracy_score

from .config import PROCESSED_DIR


LABELS = ["H", "D", "A"]
SKLEARN_LABELS = ["A", "D", "H"]


def fair_probabilities(home_odds, draw_odds, away_odds):

    raw = np.column_stack([
        1 / home_odds,
        1 / draw_odds,
        1 / away_odds,
    ])

    # Remove bookmaker overround.
    return raw / raw.sum(axis=1, keepdims=True)


def sklearn_order(probabilities):

    # H, D, A -> A, D, H

    return np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])


def brier(actual, probabilities):

    total = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(LABELS):

            observed = 1.0 if result == label else 0.0

            total += (
                probabilities[i, j]
                - observed
            ) ** 2

    return total / len(actual)


def evaluate(name, actual, probabilities):

    predicted = np.array(LABELS)[
        np.argmax(probabilities, axis=1)
    ]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    loss = log_loss(
        actual,
        sklearn_order(probabilities),
        labels=SKLEARN_LABELS
    )

    score = brier(
        actual,
        probabilities
    )

    print(f"\n{name}")
    print("-------------------------")
    print(f"Matches:     {len(actual)}")
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Brier Score: {score:.4f}")


def run():

    matches = pd.read_csv(
        PROCESSED_DIR
        / "matches_advanced_features.csv",
        parse_dates=["date"]
    )

    model = pd.read_csv(
        PROCESSED_DIR
        / "phase2_ensemble_predictions.csv",
        parse_dates=["date"]
    )

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    odds_columns = [
        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    combined = model.merge(
        matches[
            merge_columns
            + odds_columns
        ],
        on=merge_columns,
        how="left"
    )

    # ===================================
    # Require both markets so we're
    # comparing the exact same games.
    # ===================================

    required = odds_columns

    comparison = combined.dropna(
        subset=required
    ).copy()

    # Remove broken odds.
    for column in required:

        comparison = comparison[
            comparison[column] > 1
        ]

    comparison = (
        comparison
        .reset_index(drop=True)
    )

    print(
        "Matches with BOTH early "
        "and closing odds:",
        len(comparison)
    )

    actual = (
        comparison["result"]
        .to_numpy()
    )

    # ===================================
    # OUR MODEL
    # ===================================

    model_probs = comparison[
        [
            "prob_home",
            "prob_draw",
            "prob_away",
        ]
    ].to_numpy()

    evaluate(
        "OUR PHASE 2 MODEL",
        actual,
        model_probs
    )

    # ===================================
    # EARLIER MARKET
    # ===================================

    early_probs = fair_probabilities(

        comparison[
            "avg_home_odds"
        ].to_numpy(),

        comparison[
            "avg_draw_odds"
        ].to_numpy(),

        comparison[
            "avg_away_odds"
        ].to_numpy(),
    )

    evaluate(
        "EARLY / PRE-CLOSING MARKET",
        actual,
        early_probs
    )

    # ===================================
    # CLOSING MARKET
    # ===================================

    closing_probs = fair_probabilities(

        comparison[
            "avg_close_home_odds"
        ].to_numpy(),

        comparison[
            "avg_close_draw_odds"
        ].to_numpy(),

        comparison[
            "avg_close_away_odds"
        ].to_numpy(),
    )

    evaluate(
        "CLOSING MARKET",
        actual,
        closing_probs
    )

    # ===================================
    # SAVE
    # ===================================

    comparison[
        "early_prob_home"
    ] = early_probs[:, 0]

    comparison[
        "early_prob_draw"
    ] = early_probs[:, 1]

    comparison[
        "early_prob_away"
    ] = early_probs[:, 2]

    comparison[
        "close_prob_home"
    ] = closing_probs[:, 0]

    comparison[
        "close_prob_draw"
    ] = closing_probs[:, 1]

    comparison[
        "close_prob_away"
    ] = closing_probs[:, 2]

    comparison.to_csv(
        PROCESSED_DIR
        / "market_timing_comparison.csv",
        index=False
    )

    print(
        "\nSaved -> "
        "data/processed/"
        "market_timing_comparison.csv"
    )


if __name__ == "__main__":
    run()
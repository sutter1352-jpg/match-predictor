import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
)

from .config import PROCESSED_DIR


LABELS = ["H", "D", "A"]
SKLEARN_LABELS = ["A", "D", "H"]


def brier_score(actual, probabilities):

    total = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(LABELS):

            observed = (
                1.0
                if result == label
                else 0.0
            )

            total += (
                probabilities[i, j]
                - observed
            ) ** 2

    return total / len(actual)


def sklearn_order(probabilities):

    # H,D,A -> A,D,H

    return np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])


def evaluate(
    name,
    actual,
    probabilities
):

    predicted = np.array(
        LABELS
    )[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    loss = log_loss(
        actual,
        sklearn_order(
            probabilities
        ),
        labels=SKLEARN_LABELS
    )

    brier = brier_score(
        actual,
        probabilities
    )

    print(f"\n{name}")
    print("-------------------------")

    print(
        f"Matches:     {len(actual)}"
    )

    print(
        f"Accuracy:    {accuracy:.3f}"
    )

    print(
        f"Log Loss:    {loss:.4f}"
    )

    print(
        f"Brier Score: {brier:.4f}"
    )

    return accuracy, loss, brier


def bookmaker_probabilities(df):

    probabilities = []

    valid_rows = []

    for i, row in df.iterrows():

        # Prefer average market odds.

        home = row.get(
            "avg_home_odds",
            np.nan
        )

        draw = row.get(
            "avg_draw_odds",
            np.nan
        )

        away = row.get(
            "avg_away_odds",
            np.nan
        )

        # Fall back to Bet365
        # if average odds are missing.

        if (
            pd.isna(home)
            or pd.isna(draw)
            or pd.isna(away)
        ):

            home = row.get(
                "b365_home_odds",
                np.nan
            )

            draw = row.get(
                "b365_draw_odds",
                np.nan
            )

            away = row.get(
                "b365_away_odds",
                np.nan
            )

        if (
            pd.isna(home)
            or pd.isna(draw)
            or pd.isna(away)
        ):
            continue

        if (
            home <= 1
            or draw <= 1
            or away <= 1
        ):
            continue

        raw = np.array([
            1 / home,
            1 / draw,
            1 / away,
        ])

        # Remove bookmaker margin.

        normalized = (
            raw / raw.sum()
        )

        probabilities.append(
            normalized
        )

        valid_rows.append(i)

    return (
        np.array(probabilities),
        valid_rows
    )


def run():

    # ===================================
    # LOAD OUR MODEL PREDICTIONS
    # ===================================

    model = pd.read_csv(
        PROCESSED_DIR
        / "phase2_ensemble_predictions.csv",
        parse_dates=["date"]
    )

    # ===================================
    # LOAD ORIGINAL MATCH DATA
    # ===================================

    matches = pd.read_csv(
        PROCESSED_DIR
        / "matches_advanced_features.csv",
        parse_dates=["date"]
    )

    columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",

        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",

        "b365_home_odds",
        "b365_draw_odds",
        "b365_away_odds",
    ]

    available = [
        column
        for column in columns
        if column in matches.columns
    ]

    matches = matches[
        available
    ].copy()

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    combined = model.merge(
        matches,
        on=merge_columns,
        how="left"
    )

    # ===================================
    # BOOKMAKER PROBABILITIES
    # ===================================

    (
        market_probs,
        valid_rows
    ) = bookmaker_probabilities(
        combined
    )

    benchmark = combined.loc[
        valid_rows
    ].copy()

    benchmark = (
        benchmark
        .reset_index(drop=True)
    )

    print(
        "Total model matches:",
        len(combined)
    )

    print(
        "Matches with bookmaker odds:",
        len(benchmark)
    )

    # ===================================
    # OUR MODEL
    # ===================================

    model_probs = benchmark[
        [
            "prob_home",
            "prob_draw",
            "prob_away",
        ]
    ].to_numpy()

    actual = (
        benchmark["result"]
        .to_numpy()
    )

    evaluate(
        "PHASE 2 MODEL",
        actual,
        model_probs
    )

    # ===================================
    # MARKET
    # ===================================

    evaluate(
        "BOOKMAKER MARKET",
        actual,
        market_probs
    )

    # ===================================
    # SAVE COMPARISON
    # ===================================

    benchmark[
        "market_prob_home"
    ] = market_probs[:, 0]

    benchmark[
        "market_prob_draw"
    ] = market_probs[:, 1]

    benchmark[
        "market_prob_away"
    ] = market_probs[:, 2]

    benchmark[
        "model_market_home_diff"
    ] = (
        benchmark["prob_home"]
        - benchmark[
            "market_prob_home"
        ]
    )

    benchmark[
        "model_market_draw_diff"
    ] = (
        benchmark["prob_draw"]
        - benchmark[
            "market_prob_draw"
        ]
    )

    benchmark[
        "model_market_away_diff"
    ] = (
        benchmark["prob_away"]
        - benchmark[
            "market_prob_away"
        ]
    )

    output = (
        PROCESSED_DIR
        / "market_comparison.csv"
    )

    benchmark.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved -> {output}"
    )


if __name__ == "__main__":
    run()
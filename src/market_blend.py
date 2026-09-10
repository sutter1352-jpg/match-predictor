import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
)

from .config import PROCESSED_DIR


LABELS = ["H", "D", "A"]
SKLEARN_LABELS = ["A", "D", "H"]


def sklearn_order(probabilities):

    return np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])


def calculate_logloss(
    actual,
    probabilities
):

    return log_loss(
        actual,
        sklearn_order(
            probabilities
        ),
        labels=SKLEARN_LABELS
    )


def brier_score(
    actual,
    probabilities
):

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

    loss = calculate_logloss(
        actual,
        probabilities
    )

    brier = brier_score(
        actual,
        probabilities
    )

    print(f"\n{name}")
    print("-------------------------")
    print(f"Matches:     {len(actual)}")
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Brier Score: {brier:.4f}")

    return accuracy, loss, brier


def add_market_probabilities(df):

    rows = []

    for row in df.itertuples():

        # Prefer average odds.
        home = getattr(
            row,
            "avg_home_odds",
            np.nan
        )

        draw = getattr(
            row,
            "avg_draw_odds",
            np.nan
        )

        away = getattr(
            row,
            "avg_away_odds",
            np.nan
        )

        # Fall back to Bet365.
        if (
            pd.isna(home)
            or pd.isna(draw)
            or pd.isna(away)
        ):

            home = getattr(
                row,
                "b365_home_odds",
                np.nan
            )

            draw = getattr(
                row,
                "b365_draw_odds",
                np.nan
            )

            away = getattr(
                row,
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

        implied = np.array([
            1 / home,
            1 / draw,
            1 / away,
        ])

        # Remove bookmaker margin.
        implied = (
            implied
            / implied.sum()
        )

        data = row._asdict()

        data[
            "market_prob_home"
        ] = implied[0]

        data[
            "market_prob_draw"
        ] = implied[1]

        data[
            "market_prob_away"
        ] = implied[2]

        rows.append(data)

    return pd.DataFrame(rows)


def prepare(
    predictions,
    matches
):

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

    columns = [
        c
        for c in columns
        if c in matches.columns
    ]

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    combined = predictions.merge(
        matches[columns],
        on=merge_columns,
        how="left"
    )

    return add_market_probabilities(
        combined
    )


def model_probs(df):

    return df[
        [
            "prob_home",
            "prob_draw",
            "prob_away",
        ]
    ].to_numpy()


def market_probs(df):

    return df[
        [
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
    ].to_numpy()


def run():

    matches = pd.read_csv(
        PROCESSED_DIR
        / "matches_advanced_features.csv",
        parse_dates=["date"]
    )

    validation_predictions = pd.read_csv(
        PROCESSED_DIR
        / "phase2_validation_predictions.csv",
        parse_dates=["date"]
    )

    test_predictions = pd.read_csv(
        PROCESSED_DIR
        / "phase2_ensemble_predictions.csv",
        parse_dates=["date"]
    )

    validation = prepare(
        validation_predictions,
        matches
    )

    test = prepare(
        test_predictions,
        matches
    )

    print(
        "Validation matches:",
        len(validation)
    )

    print(
        "Test matches:",
        len(test)
    )

    model_validation = model_probs(
        validation
    )

    market_validation = market_probs(
        validation
    )

    model_test = model_probs(
        test
    )

    market_test = market_probs(
        test
    )

    actual_validation = (
        validation["result"]
        .to_numpy()
    )

    actual_test = (
        test["result"]
        .to_numpy()
    )

    # ====================================
    # FIND BEST MARKET / MODEL BLEND
    # ====================================

    best_weight = None
    best_loss = float("inf")

    for model_weight in np.linspace(
        0,
        1,
        101
    ):

        market_weight = (
            1 - model_weight
        )

        blended = (
            model_weight
            * model_validation

            +

            market_weight
            * market_validation
        )

        loss = calculate_logloss(
            actual_validation,
            blended
        )

        if loss < best_loss:

            best_loss = loss
            best_weight = model_weight

    market_weight = (
        1 - best_weight
    )

    print(
        "\nBEST VALIDATION BLEND"
    )

    print(
        "-------------------------"
    )

    print(
        f"Our model: {best_weight:.0%}"
    )

    print(
        f"Market:    {market_weight:.0%}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_loss:.4f}"
    )

    # ====================================
    # VALIDATION
    # ====================================

    evaluate(
        "MARKET VALIDATION",
        actual_validation,
        market_validation
    )

    validation_blend = (
        best_weight
        * model_validation

        +

        market_weight
        * market_validation
    )

    evaluate(
        "BLENDED VALIDATION",
        actual_validation,
        validation_blend
    )

    # ====================================
    # HELD-OUT TEST
    # ====================================

    evaluate(
        "MARKET TEST",
        actual_test,
        market_test
    )

    evaluate(
        "OUR MODEL TEST",
        actual_test,
        model_test
    )

    test_blend = (
        best_weight
        * model_test

        +

        market_weight
        * market_test
    )

    evaluate(
        "BLENDED TEST",
        actual_test,
        test_blend
    )

    # ====================================
    # SAVE DISAGREEMENTS
    # ====================================

    output = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    output[
        "model_prob_home"
    ] = model_test[:, 0]

    output[
        "model_prob_draw"
    ] = model_test[:, 1]

    output[
        "model_prob_away"
    ] = model_test[:, 2]

    output[
        "market_prob_home"
    ] = market_test[:, 0]

    output[
        "market_prob_draw"
    ] = market_test[:, 1]

    output[
        "market_prob_away"
    ] = market_test[:, 2]

    output[
        "home_edge"
    ] = (
        model_test[:, 0]
        - market_test[:, 0]
    )

    output[
        "draw_edge"
    ] = (
        model_test[:, 1]
        - market_test[:, 1]
    )

    output[
        "away_edge"
    ] = (
        model_test[:, 2]
        - market_test[:, 2]
    )

    output.to_csv(
        PROCESSED_DIR
        / "model_market_disagreements.csv",
        index=False
    )


if __name__ == "__main__":
    run()
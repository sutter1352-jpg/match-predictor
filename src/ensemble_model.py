import pandas as pd
import numpy as np

from sklearn.metrics import (
    log_loss,
    accuracy_score,
)

from .config import PROCESSED_DIR


# Our internal probability order
OUR_LABELS = ["H", "D", "A"]

# sklearn expects alphabetical order
SKLEARN_LABELS = ["A", "D", "H"]


def brier_score(actual, probabilities):

    score = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(OUR_LABELS):

            observed = (
                1.0
                if result == label
                else 0.0
            )

            score += (
                probabilities[i, j]
                - observed
            ) ** 2

    return score / len(actual)


def convert_for_log_loss(probabilities):
    """
    Our probability columns are:

        H, D, A

    sklearn log_loss expects:

        A, D, H

    So reorder the columns.
    """

    return np.column_stack([
        probabilities[:, 2],  # Away
        probabilities[:, 1],  # Draw
        probabilities[:, 0],  # Home
    ])


def evaluate(
    name,
    df,
    probabilities
):

    actual = (
        df["result"]
        .to_numpy()
    )

    labels = np.array(
        OUR_LABELS
    )

    predicted = labels[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    accuracy = accuracy_score(
        actual,
        predicted
    )

    logloss_probabilities = (
        convert_for_log_loss(
            probabilities
        )
    )

    loss = log_loss(
        actual,
        logloss_probabilities,
        labels=SKLEARN_LABELS
    )

    brier = brier_score(
        actual,
        probabilities
    )

    print(f"\n{name}")
    print("-------------------------")

    print(
        f"Matches:     {len(df)}"
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


def get_probabilities(
    df,
    prefix
):

    return np.column_stack([
        df[f"prob_home_{prefix}"],
        df[f"prob_draw_{prefix}"],
        df[f"prob_away_{prefix}"],
    ])


def run():

    logistic_path = (
        PROCESSED_DIR
        / "walk_forward_logistic.csv"
    )

    poisson_path = (
        PROCESSED_DIR
        / "walk_forward_poisson.csv"
    )

    logistic = pd.read_csv(
        logistic_path,
        parse_dates=["date"]
    )

    poisson = pd.read_csv(
        poisson_path,
        parse_dates=["date"]
    )

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    merged = logistic.merge(
        poisson,
        on=merge_columns,
        suffixes=(
            "_logistic",
            "_poisson"
        )
    )

    merged = (
        merged
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "Matched predictions:",
        len(merged)
    )

    # =================================
    # VALIDATION: 2024/25
    # =================================

    validation = merged[
        (
            merged["date"]
            >= pd.Timestamp(
                "2024-08-01"
            )
        )
        &
        (
            merged["date"]
            < pd.Timestamp(
                "2025-08-01"
            )
        )
    ].copy()

    # =================================
    # TEST: 2025/26
    # =================================

    test = merged[
        merged["date"]
        >= pd.Timestamp(
            "2025-08-01"
        )
    ].copy()

    print(
        "Validation matches:",
        len(validation)
    )

    print(
        "Test matches:",
        len(test)
    )

    validation_logistic = (
        get_probabilities(
            validation,
            "logistic"
        )
    )

    validation_poisson = (
        get_probabilities(
            validation,
            "poisson"
        )
    )

    # =================================
    # FIND BEST ENSEMBLE WEIGHT
    # =================================

    best_weight = None
    best_loss = float("inf")

    print(
        "\nSearching ensemble weights..."
    )

    for weight in np.linspace(
        0,
        1,
        101
    ):

        blended = (
            weight
            * validation_logistic
            +
            (
                1 - weight
            )
            * validation_poisson
        )

        logloss_probs = (
            convert_for_log_loss(
                blended
            )
        )

        loss = log_loss(
            validation["result"],
            logloss_probs,
            labels=SKLEARN_LABELS
        )

        if loss < best_loss:

            best_loss = loss
            best_weight = weight

    poisson_weight = (
        1 - best_weight
    )

    print(
        "\nBEST VALIDATION WEIGHTS"
    )

    print(
        "-------------------------"
    )

    print(
        f"Logistic: "
        f"{best_weight:.0%}"
    )

    print(
        f"Poisson:  "
        f"{poisson_weight:.0%}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_loss:.4f}"
    )

    # =================================
    # VALIDATION
    # =================================

    validation_blended = (
        best_weight
        * validation_logistic
        +
        poisson_weight
        * validation_poisson
    )

    evaluate(
        "ENSEMBLE VALIDATION",
        validation,
        validation_blended
    )

    # =================================
    # TEST
    # =================================

    test_logistic = (
        get_probabilities(
            test,
            "logistic"
        )
    )

    test_poisson = (
        get_probabilities(
            test,
            "poisson"
        )
    )

    test_blended = (
        best_weight
        * test_logistic
        +
        poisson_weight
        * test_poisson
    )

    evaluate(
        "LOGISTIC TEST",
        test,
        test_logistic
    )

    evaluate(
        "POISSON TEST",
        test,
        test_poisson
    )

    evaluate(
        "ENSEMBLE TEST",
        test,
        test_blended
    )

    # =================================
    # SAVE FINAL TEST PREDICTIONS
    # =================================

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
        "prob_home"
    ] = test_blended[:, 0]

    output[
        "prob_draw"
    ] = test_blended[:, 1]

    output[
        "prob_away"
    ] = test_blended[:, 2]

    labels = np.array(
        OUR_LABELS
    )

    output[
        "prediction"
    ] = labels[
        np.argmax(
            test_blended,
            axis=1
        )
    ]

    output[
        "logistic_weight"
    ] = best_weight

    output[
        "poisson_weight"
    ] = poisson_weight

    output_path = (
        PROCESSED_DIR
        / "ensemble_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved -> {output_path}"
    )


if __name__ == "__main__":
    run()
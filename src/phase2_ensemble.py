import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    accuracy_score,
)

from .config import PROCESSED_DIR


LABELS = ["H", "D", "A"]
SKLEARN_LABELS = ["A", "D", "H"]


def get_probs(df):
    return df[
        [
            "prob_home",
            "prob_draw",
            "prob_away"
        ]
    ].to_numpy()


def sklearn_order(probabilities):
    """
    Convert H,D,A -> A,D,H
    for sklearn log_loss.
    """

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

    score = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(LABELS):

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


def evaluate(
    name,
    actual,
    probabilities
):

    label_array = np.array(
        LABELS
    )

    predicted = label_array[
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

    print(
        f"\n{name}"
    )

    print(
        "-------------------------"
    )

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

    return (
        accuracy,
        loss,
        brier
    )


def temperature_scale(
    probabilities,
    temperature
):
    """
    T > 1 makes probabilities
    less confident.

    T < 1 makes probabilities
    more confident.
    """

    probabilities = np.clip(
        probabilities,
        1e-9,
        1.0
    )

    scaled = (
        probabilities
        ** (1.0 / temperature)
    )

    scaled = (
        scaled
        / scaled.sum(
            axis=1,
            keepdims=True
        )
    )

    return scaled


def find_temperature(
    actual,
    probabilities
):

    best_temperature = None
    best_loss = float("inf")

    for temperature in np.arange(
        0.50,
        3.01,
        0.01
    ):

        scaled = temperature_scale(
            probabilities,
            temperature
        )

        loss = calculate_logloss(
            actual,
            scaled
        )

        if loss < best_loss:

            best_loss = loss
            best_temperature = temperature

    return (
        best_temperature,
        best_loss
    )


def merge_models(
    logistic,
    poisson,
    boosting
):

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    logistic = logistic.rename(
        columns={
            "prob_home":
                "prob_home_logistic",

            "prob_draw":
                "prob_draw_logistic",

            "prob_away":
                "prob_away_logistic",
        }
    )

    poisson = poisson.rename(
        columns={
            "prob_home":
                "prob_home_poisson",

            "prob_draw":
                "prob_draw_poisson",

            "prob_away":
                "prob_away_poisson",
        }
    )

    boosting = boosting.rename(
        columns={
            "prob_home":
                "prob_home_boosting",

            "prob_draw":
                "prob_draw_boosting",

            "prob_away":
                "prob_away_boosting",
        }
    )

    merged = logistic.merge(
        poisson,
        on=merge_columns
    )

    merged = merged.merge(
        boosting,
        on=merge_columns
    )

    return merged


def model_probs(
    df,
    model
):

    return np.column_stack([
        df[
            f"prob_home_{model}"
        ],

        df[
            f"prob_draw_{model}"
        ],

        df[
            f"prob_away_{model}"
        ],
    ])


def run():

    # =====================================
    # LOAD PHASE 1 WALK-FORWARD RESULTS
    # =====================================

    logistic = pd.read_csv(
        PROCESSED_DIR
        / "walk_forward_logistic.csv",
        parse_dates=["date"]
    )

    poisson = pd.read_csv(
        PROCESSED_DIR
        / "walk_forward_poisson.csv",
        parse_dates=["date"]
    )

    boosting_validation = pd.read_csv(
        PROCESSED_DIR
        / "boosting_validation_predictions.csv",
        parse_dates=["date"]
    )

    boosting_test = pd.read_csv(
        PROCESSED_DIR
        / "boosting_test_predictions.csv",
        parse_dates=["date"]
    )

    # =====================================
    # SPLIT LOGISTIC / POISSON
    # =====================================

    logistic_validation = logistic[
        (
            logistic["date"]
            >= pd.Timestamp(
                "2024-08-01"
            )
        )
        &
        (
            logistic["date"]
            < pd.Timestamp(
                "2025-08-01"
            )
        )
    ].copy()

    logistic_test = logistic[
        logistic["date"]
        >= pd.Timestamp(
            "2025-08-01"
        )
    ].copy()

    poisson_validation = poisson[
        (
            poisson["date"]
            >= pd.Timestamp(
                "2024-08-01"
            )
        )
        &
        (
            poisson["date"]
            < pd.Timestamp(
                "2025-08-01"
            )
        )
    ].copy()

    poisson_test = poisson[
        poisson["date"]
        >= pd.Timestamp(
            "2025-08-01"
        )
    ].copy()

    # =====================================
    # MERGE
    # =====================================

    validation = merge_models(
        logistic_validation,
        poisson_validation,
        boosting_validation
    )

    test = merge_models(
        logistic_test,
        poisson_test,
        boosting_test
    )

    print(
        "Validation matches:",
        len(validation)
    )

    print(
        "Test matches:",
        len(test)
    )

    # =====================================
    # GET PROBABILITIES
    # =====================================

    log_val = model_probs(
        validation,
        "logistic"
    )

    poi_val = model_probs(
        validation,
        "poisson"
    )

    boost_val = model_probs(
        validation,
        "boosting"
    )

    log_test = model_probs(
        test,
        "logistic"
    )

    poi_test = model_probs(
        test,
        "poisson"
    )

    boost_test = model_probs(
        test,
        "boosting"
    )

    actual_validation = (
        validation["result"]
        .to_numpy()
    )

    actual_test = (
        test["result"]
        .to_numpy()
    )

    # =====================================
    # CALIBRATE BOOSTING
    # =====================================

    temperature, temp_loss = (
        find_temperature(
            actual_validation,
            boost_val
        )
    )

    print(
        "\nBOOSTING CALIBRATION"
    )

    print(
        "-------------------------"
    )

    print(
        f"Best temperature: "
        f"{temperature:.2f}"
    )

    print(
        f"Validation Log Loss: "
        f"{temp_loss:.4f}"
    )

    calibrated_boost_val = (
        temperature_scale(
            boost_val,
            temperature
        )
    )

    calibrated_boost_test = (
        temperature_scale(
            boost_test,
            temperature
        )
    )

    evaluate(
        "CALIBRATED BOOSTING VALIDATION",
        actual_validation,
        calibrated_boost_val
    )

    evaluate(
        "CALIBRATED BOOSTING TEST",
        actual_test,
        calibrated_boost_test
    )

    # =====================================
    # SEARCH 3-MODEL ENSEMBLE
    # =====================================

    best_loss = float("inf")

    best_log_weight = None
    best_poi_weight = None
    best_boost_weight = None

    # Search in 5% increments.

    for log_weight in np.arange(
        0,
        1.01,
        0.05
    ):

        for poi_weight in np.arange(
            0,
            1.01 - log_weight,
            0.05
        ):

            boost_weight = (
                1
                - log_weight
                - poi_weight
            )

            if boost_weight < 0:
                continue

            blended = (
                log_weight
                * log_val

                +

                poi_weight
                * poi_val

                +

                boost_weight
                * calibrated_boost_val
            )

            loss = calculate_logloss(
                actual_validation,
                blended
            )

            if loss < best_loss:

                best_loss = loss

                best_log_weight = (
                    log_weight
                )

                best_poi_weight = (
                    poi_weight
                )

                best_boost_weight = (
                    boost_weight
                )

    print(
        "\nBEST PHASE 2 WEIGHTS"
    )

    print(
        "-------------------------"
    )

    print(
        f"Logistic: "
        f"{best_log_weight:.0%}"
    )

    print(
        f"Poisson:  "
        f"{best_poi_weight:.0%}"
    )

    print(
        f"Boosting: "
        f"{best_boost_weight:.0%}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_loss:.4f}"
    )

    # =====================================
    # FINAL VALIDATION ENSEMBLE
    # =====================================

    validation_ensemble = (

        best_log_weight
        * log_val

        +

        best_poi_weight
        * poi_val

        +

        best_boost_weight
        * calibrated_boost_val
    )

    evaluate(
        "PHASE 2 ENSEMBLE VALIDATION",
        actual_validation,
        validation_ensemble
    )

    # =====================================
    # FINAL HELD-OUT TEST
    # =====================================

    test_ensemble = (

        best_log_weight
        * log_test

        +

        best_poi_weight
        * poi_test

        +

        best_boost_weight
        * calibrated_boost_test
    )

    evaluate(
        "PHASE 2 ENSEMBLE TEST",
        actual_test,
        test_ensemble
    )

    # =====================================
    # SAVE
    # =====================================

    output = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    output["prob_home"] = (
        test_ensemble[:, 0]
    )

    output["prob_draw"] = (
        test_ensemble[:, 1]
    )

    output["prob_away"] = (
        test_ensemble[:, 2]
    )

    labels = np.array(
        LABELS
    )

    output["prediction"] = labels[
        np.argmax(
            test_ensemble,
            axis=1
        )
    ]

    output["logistic_weight"] = (
        best_log_weight
    )

    output["poisson_weight"] = (
        best_poi_weight
    )

    output["boosting_weight"] = (
        best_boost_weight
    )

    output[
        "boosting_temperature"
    ] = temperature

    output_path = (
        PROCESSED_DIR
        / "phase2_ensemble_predictions.csv"
    )
    # =====================================
    # SAVE VALIDATION ENSEMBLE
    # =====================================

    validation_output = validation[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    validation_output[
        "prob_home"
    ] = validation_ensemble[:, 0]

    validation_output[
        "prob_draw"
    ] = validation_ensemble[:, 1]

    validation_output[
        "prob_away"
    ] = validation_ensemble[:, 2]

    validation_path = (
        PROCESSED_DIR
        / "phase2_validation_predictions.csv"
    )

    validation_output.to_csv(
        validation_path,
        index=False
    )

    print(
        f"\nSaved validation -> {validation_path}"
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
import json
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR


CALIBRATION_END = pd.Timestamp("2025-08-01")
TEST_END = pd.Timestamp("2026-08-01")


def temperature_scale(
    probabilities,
    temperature
):

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    logits = np.log(
        probabilities
    )

    logits = (
        logits
        / temperature
    )

    logits = (
        logits
        - logits.max(
            axis=1,
            keepdims=True
        )
    )

    exp_logits = np.exp(
        logits
    )

    return (
        exp_logits
        / exp_logits.sum(
            axis=1,
            keepdims=True
        )
    )


def log_loss(
    actual,
    probabilities
):

    mapping = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    indexes = np.array([
        mapping[x]
        for x in actual
    ])

    selected = probabilities[
        np.arange(
            len(probabilities)
        ),
        indexes
    ]

    return -np.mean(
        np.log(
            np.clip(
                selected,
                1e-12,
                1
            )
        )
    )


def accuracy(
    actual,
    probabilities
):

    labels = np.array([
        "H",
        "D",
        "A",
    ])

    prediction = labels[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    return (
        prediction
        == actual
    ).mean()


def calibration_table(
    actual,
    probabilities
):

    labels = np.array([
        "H",
        "D",
        "A",
    ])

    predicted_index = np.argmax(
        probabilities,
        axis=1
    )

    predicted_result = labels[
        predicted_index
    ]

    confidence = np.max(
        probabilities,
        axis=1
    )

    correct = (
        predicted_result
        == actual
    )

    df = pd.DataFrame({
        "confidence": confidence,
        "correct": correct,
    })

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

    names = [
        "33-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    df["bin"] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=names,
        right=False
    )

    summary = (
        df
        .groupby(
            "bin",
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

    summary["gap"] = (
        summary[
            "actual_accuracy"
        ]
        -
        summary[
            "average_prediction"
        ]
    )

    return summary


def print_table(
    table
):

    display = table.copy()

    for column in [
        "average_prediction",
        "actual_accuracy",
        "gap",
    ]:

        display[column] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "average_prediction":
                    lambda x:
                    f"{x:.1f}%",

                "actual_accuracy":
                    lambda x:
                    f"{x:.1f}%",

                "gap":
                    lambda x:
                    f"{x:+.1f}%",
            }
        )
    )


def run():

    path = (
        PROCESSED_DIR
        /
        "match_predictor_backtest_predictions.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
    ]

    calibration = df[
        df["date"]
        < CALIBRATION_END
    ].copy()

    test = df[
        (
            df["date"]
            >= CALIBRATION_END
        )
        &
        (
            df["date"]
            < TEST_END
        )
    ].copy()

    calibration_probs = calibration[
        probability_columns
    ].to_numpy()

    calibration_actual = calibration[
        "actual_result"
    ].to_numpy()

    # ========================================================
    # SEARCH TEMPERATURE
    # ========================================================

    temperatures = np.arange(
        0.70,
        2.01,
        0.01
    )

    best_temperature = None
    best_loss = float("inf")

    print(
        "\nSEARCHING FOR BEST TEMPERATURE"
    )

    print(
        "=" * 60
    )

    for temperature in temperatures:

        scaled = temperature_scale(
            calibration_probs,
            temperature
        )

        loss = log_loss(
            calibration_actual,
            scaled
        )

        if loss < best_loss:

            best_loss = loss
            best_temperature = (
                float(temperature)
            )

    print(
        f"Best temperature: "
        f"{best_temperature:.2f}"
    )

    print(
        f"Calibration Log Loss: "
        f"{best_loss:.4f}"
    )

    # ========================================================
    # STRICT 2025/26 TEST
    # ========================================================

    test_probs = test[
        probability_columns
    ].to_numpy()

    actual = test[
        "actual_result"
    ].to_numpy()

    calibrated_probs = (
        temperature_scale(
            test_probs,
            best_temperature
        )
    )

    raw_loss = log_loss(
        actual,
        test_probs
    )

    calibrated_loss = log_loss(
        actual,
        calibrated_probs
    )

    raw_accuracy = accuracy(
        actual,
        test_probs
    )

    calibrated_accuracy = accuracy(
        actual,
        calibrated_probs
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "2025/26 STRICT HOLDOUT"
    )

    print(
        "=" * 60
    )

    print(
        f"Matches: "
        f"{len(test):,}"
    )

    print(
        "\nRAW MODEL"
    )

    print(
        f"Accuracy: "
        f"{raw_accuracy:.2%}"
    )

    print(
        f"Log Loss: "
        f"{raw_loss:.4f}"
    )

    print(
        "\nCALIBRATED MODEL"
    )

    print(
        f"Accuracy: "
        f"{calibrated_accuracy:.2%}"
    )

    print(
        f"Log Loss: "
        f"{calibrated_loss:.4f}"
    )

    print(
        f"\nLog Loss improvement: "
        f"{raw_loss - calibrated_loss:+.4f}"
    )

    # ========================================================
    # CALIBRATION TABLE
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RAW CALIBRATION"
    )

    print(
        "=" * 60
    )

    raw_table = calibration_table(
        actual,
        test_probs
    )

    print_table(
        raw_table
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CALIBRATED PROBABILITIES"
    )

    print(
        "=" * 60
    )

    calibrated_table = (
        calibration_table(
            actual,
            calibrated_probs
        )
    )

    print_table(
        calibrated_table
    )

    # ========================================================
    # SAVE TEMPERATURE
    # ========================================================

    output = {
        "temperature":
            best_temperature,

        "calibration_log_loss":
            float(
                best_loss
            ),

        "holdout_raw_log_loss":
            float(
                raw_loss
            ),

        "holdout_calibrated_log_loss":
            float(
                calibrated_loss
            ),
    }

    output_path = (
        PROCESSED_DIR
        /
        "match_predictor_calibration.json"
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    run()
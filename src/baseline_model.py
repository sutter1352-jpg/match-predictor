import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from .config import PROCESSED_DIR


FEATURES = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_home_adjusted",

    "home_points_last5",
    "away_points_last5",

    "home_goals_for_last5",
    "away_goals_for_last5",

    "home_goals_against_last5",
    "away_goals_against_last5",
]


def multiclass_brier(y_true, probabilities, classes):
    score = 0

    for i, actual in enumerate(y_true):
        for j, class_name in enumerate(classes):
            observed = 1 if actual == class_name else 0
            score += (probabilities[i][j] - observed) ** 2

    return score / len(y_true)


def run():

    path = PROCESSED_DIR / "matches_with_features.csv"

    df = pd.read_csv(path, parse_dates=["date"])

    df = df.dropna(subset=FEATURES + ["result"]).copy()

    # -------------------------
    # TIME-BASED SPLIT
    # -------------------------

    # Train:
    # 2015/16 through 2023/24
    train = df[
        df["season"].astype(str).isin([
            "1516",
            "1617",
            "1718",
            "1819",
            "1920",
            "2021",
            "2122",
            "2223",
            "2324",
        ])
    ]

    # Validation:
    # 2024/25
    validation = df[
        df["season"].astype(str) == "2425"
    ]

    # Test:
    # 2025/26
    test = df[
        df["season"].astype(str) == "2526"
    ]

    print("Training matches:", len(train))
    print("Validation matches:", len(validation))
    print("Test matches:", len(test))

    X_train = train[FEATURES]
    y_train = train["result"]

    X_validation = validation[FEATURES]
    y_validation = validation["result"]

    X_test = test[FEATURES]
    y_test = test["result"]

    # -------------------------
    # MODEL
    # -------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        ),
    ])

    model.fit(X_train, y_train)

    # -------------------------
    # VALIDATION
    # -------------------------

    validation_probabilities = model.predict_proba(X_validation)
    validation_predictions = model.predict(X_validation)

    classes = model.classes_

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_logloss = log_loss(
        y_validation,
        validation_probabilities,
        labels=classes
    )

    validation_brier = multiclass_brier(
        y_validation.to_numpy(),
        validation_probabilities,
        classes
    )

    print("\nVALIDATION RESULTS")
    print("-------------------")

    print(
        f"Accuracy:   {validation_accuracy:.3f}"
    )

    print(
        f"Log Loss:   {validation_logloss:.3f}"
    )

    print(
        f"Brier Score:{validation_brier:.3f}"
    )

    # -------------------------
    # FINAL TEST
    # -------------------------

    test_probabilities = model.predict_proba(X_test)
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_logloss = log_loss(
        y_test,
        test_probabilities,
        labels=classes
    )

    test_brier = multiclass_brier(
        y_test.to_numpy(),
        test_probabilities,
        classes
    )

    print("\nTEST RESULTS")
    print("-------------------")

    print(
        f"Accuracy:   {test_accuracy:.3f}"
    )

    print(
        f"Log Loss:   {test_logloss:.3f}"
    )

    print(
        f"Brier Score:{test_brier:.3f}"
    )

    # -------------------------
    # SAVE PREDICTIONS
    # -------------------------

    predictions = test[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    class_to_index = {
        class_name: i
        for i, class_name in enumerate(classes)
    }

    predictions["prob_home"] = (
        test_probabilities[:, class_to_index["H"]]
    )

    predictions["prob_draw"] = (
        test_probabilities[:, class_to_index["D"]]
    )

    predictions["prob_away"] = (
        test_probabilities[:, class_to_index["A"]]
    )

    predictions["prediction"] = test_predictions

    output = (
        PROCESSED_DIR
        / "baseline_predictions.csv"
    )

    predictions.to_csv(
        output,
        index=False
    )

    print(
        f"\nPredictions saved -> {output}"
    )

    print("\nExample predictions:\n")

    print(
        predictions[
            [
                "home_team",
                "away_team",
                "prob_home",
                "prob_draw",
                "prob_away",
                "result",
            ]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    run()
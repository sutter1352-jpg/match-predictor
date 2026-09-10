import pandas as pd
import numpy as np

from scipy.stats import poisson

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, accuracy_score

from .config import PROCESSED_DIR


MAX_GOALS = 10


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


def multiclass_brier(y_true, probabilities):

    class_order = ["H", "D", "A"]

    total = 0.0

    for i, actual in enumerate(y_true):

        for j, label in enumerate(class_order):

            observed = (
                1.0
                if actual == label
                else 0.0
            )

            total += (
                probabilities[i, j]
                - observed
            ) ** 2

    return total / len(y_true)


# --------------------------------
# LOGISTIC MODEL
# --------------------------------

def build_logistic():

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

    return model


# --------------------------------
# POISSON MODEL
# --------------------------------

def build_poisson():

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            PoissonRegressor(
                alpha=0.1,
                max_iter=2000
            )
        ),

    ])

    return model


def score_probabilities(
    expected_home,
    expected_away
):

    home = 0.0
    draw = 0.0
    away = 0.0

    for hg in range(MAX_GOALS + 1):

        p_home_goals = poisson.pmf(
            hg,
            expected_home
        )

        for ag in range(MAX_GOALS + 1):

            p_away_goals = poisson.pmf(
                ag,
                expected_away
            )

            probability = (
                p_home_goals
                * p_away_goals
            )

            if hg > ag:
                home += probability

            elif hg == ag:
                draw += probability

            else:
                away += probability

    total = home + draw + away

    return [
        home / total,
        draw / total,
        away / total
    ]


# --------------------------------
# WALK FORWARD
# --------------------------------

def walk_forward(df, start_date):

    future = df[
        df["date"] >= start_date
    ].copy()

    # Predict one calendar month at a time.
    future["prediction_month"] = (
        future["date"]
        .dt.to_period("M")
    )

    logistic_results = []
    poisson_results = []

    months = sorted(
        future["prediction_month"].unique()
    )

    for month in months:

        current = future[
            future["prediction_month"]
            == month
        ].copy()

        prediction_date = (
            current["date"].min()
        )

        # CRITICAL:
        # Train only on matches that happened
        # before this prediction period.
        train = df[
            df["date"] < prediction_date
        ].copy()

        train = train.dropna(
            subset=
            FEATURES
            + [
                "result",
                "home_goals",
                "away_goals"
            ]
        )

        current = current.dropna(
            subset=
            FEATURES
            + ["result"]
        )

        if len(current) == 0:
            continue

        print(
            f"\nPredicting {month}"
        )

        print(
            f"Training games: {len(train)}"
        )

        print(
            f"Games predicted: {len(current)}"
        )

        # =============================
        # LOGISTIC
        # =============================

        logistic = build_logistic()

        logistic.fit(
            train[FEATURES],
            train["result"]
        )

        probs = logistic.predict_proba(
            current[FEATURES]
        )

        classes = logistic.classes_

        index = {
            label: i
            for i, label in enumerate(classes)
        }

        logistic_probs = np.column_stack([

            probs[:, index["H"]],
            probs[:, index["D"]],
            probs[:, index["A"]],

        ])

        for i, row in enumerate(
            current.itertuples()
        ):

            logistic_results.append({

                "date":
                    row.date,

                "league":
                    row.league,

                "home_team":
                    row.home_team,

                "away_team":
                    row.away_team,

                "prob_home":
                    logistic_probs[i, 0],

                "prob_draw":
                    logistic_probs[i, 1],

                "prob_away":
                    logistic_probs[i, 2],

                "result":
                    row.result,
            })

        # =============================
        # POISSON
        # =============================

        home_model = build_poisson()

        away_model = build_poisson()

        home_model.fit(
            train[FEATURES],
            train["home_goals"]
        )

        away_model.fit(
            train[FEATURES],
            train["away_goals"]
        )

        expected_home = (
            home_model.predict(
                current[FEATURES]
            )
        )

        expected_away = (
            away_model.predict(
                current[FEATURES]
            )
        )

        for i, row in enumerate(
            current.itertuples()
        ):

            probabilities = (
                score_probabilities(
                    expected_home[i],
                    expected_away[i]
                )
            )

            poisson_results.append({

                "date":
                    row.date,

                "league":
                    row.league,

                "home_team":
                    row.home_team,

                "away_team":
                    row.away_team,

                "expected_home_goals":
                    expected_home[i],

                "expected_away_goals":
                    expected_away[i],

                "prob_home":
                    probabilities[0],

                "prob_draw":
                    probabilities[1],

                "prob_away":
                    probabilities[2],

                "result":
                    row.result,
            })

    return (
        pd.DataFrame(logistic_results),
        pd.DataFrame(poisson_results)
    )


# --------------------------------
# EVALUATE
# --------------------------------

def evaluate(name, predictions):

    probabilities = predictions[
        [
            "prob_home",
            "prob_draw",
            "prob_away"
        ]
    ].to_numpy()

    actual = (
        predictions["result"]
        .to_numpy()
    )

    labels = np.array(
        ["H", "D", "A"]
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

    # sklearn labels order:
    # A, D, H

    logloss_probabilities = (
        np.column_stack([
            probabilities[:, 2],
            probabilities[:, 1],
            probabilities[:, 0],
        ])
    )

    loss = log_loss(
        actual,
        logloss_probabilities,
        labels=["A", "D", "H"]
    )

    brier = multiclass_brier(
        actual,
        probabilities
    )

    print(
        f"\n{name}"
    )

    print(
        "-----------------------"
    )

    print(
        f"Matches:     {len(predictions)}"
    )

    print(
        f"Accuracy:    {accuracy:.3f}"
    )

    print(
        f"Log Loss:    {loss:.3f}"
    )

    print(
        f"Brier Score: {brier:.3f}"
    )

    return (
        accuracy,
        loss,
        brier
    )


def run():

    path = (
        PROCESSED_DIR
        / "matches_with_features.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    df = df.sort_values(
        "date"
    ).copy()

    # Start walk-forward simulation
    # with the 2024/25 season.

    start_date = pd.Timestamp(
        "2024-08-01"
    )

    (
        logistic_predictions,
        poisson_predictions
    ) = walk_forward(
        df,
        start_date
    )

    evaluate(
        "WALK-FORWARD LOGISTIC",
        logistic_predictions
    )

    evaluate(
        "WALK-FORWARD POISSON",
        poisson_predictions
    )

    logistic_output = (
        PROCESSED_DIR
        / "walk_forward_logistic.csv"
    )

    poisson_output = (
        PROCESSED_DIR
        / "walk_forward_poisson.csv"
    )

    logistic_predictions.to_csv(
        logistic_output,
        index=False
    )

    poisson_predictions.to_csv(
        poisson_output,
        index=False
    )

    print(
        "\nSaved predictions:"
    )

    print(
        logistic_output
    )

    print(
        poisson_output
    )


if __name__ == "__main__":
    run()
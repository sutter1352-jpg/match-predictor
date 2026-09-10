import pandas as pd
import numpy as np

from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

from .config import PROCESSED_DIR


MAX_GOALS = 8


def prepare_goal_data(df):
    """
    Convert every match into two observations:

    home team scoring goals
    away team scoring goals
    """

    home = pd.DataFrame({
        "team": df["home_team"],
        "opponent": df["away_team"],
        "league": df["league"],
        "home": 1,
        "goals": df["home_goals"],
    })

    away = pd.DataFrame({
        "team": df["away_team"],
        "opponent": df["home_team"],
        "league": df["league"],
        "home": 0,
        "goals": df["away_goals"],
    })

    return pd.concat(
        [home, away],
        ignore_index=True
    )


def build_model():

    categorical = [
        "team",
        "opponent",
        "league",
    ]

    numeric = [
        "home"
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        ),
        (
            "numeric",
            "passthrough",
            numeric
        ),
    ])

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            PoissonRegressor(
                alpha=0.1,
                max_iter=1000
            )
        ),
    ])

    return model


def calculate_match_probabilities(
    expected_home,
    expected_away
):

    home_win = 0
    draw = 0
    away_win = 0

    for home_goals in range(MAX_GOALS + 1):

        p_home = poisson.pmf(
            home_goals,
            expected_home
        )

        for away_goals in range(MAX_GOALS + 1):

            p_away = poisson.pmf(
                away_goals,
                expected_away
            )

            probability = (
                p_home * p_away
            )

            if home_goals > away_goals:
                home_win += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away_win += probability

    total = home_win + draw + away_win

    return (
        home_win / total,
        draw / total,
        away_win / total,
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

    train_seasons = [
        "1516",
        "1617",
        "1718",
        "1819",
        "1920",
        "2021",
        "2122",
        "2223",
        "2324",
    ]

    train = df[
        df["season"]
        .astype(str)
        .isin(train_seasons)
    ].copy()

    validation = df[
        df["season"]
        .astype(str)
        == "2425"
    ].copy()

    test = df[
        df["season"]
        .astype(str)
        == "2526"
    ].copy()

    print(
        "Training matches:",
        len(train)
    )

    print(
        "Validation matches:",
        len(validation)
    )

    print(
        "Test matches:",
        len(test)
    )

    # --------------------
    # TRAIN GOAL MODEL
    # --------------------

    goal_train = prepare_goal_data(train)

    model = build_model()

    X_train = goal_train[
        [
            "team",
            "opponent",
            "league",
            "home",
        ]
    ]

    y_train = goal_train["goals"]

    model.fit(
        X_train,
        y_train
    )

    # --------------------
    # PREDICT MATCHES
    # --------------------

    def predict_matches(matches):

        predictions = []

        for row in matches.itertuples():

            home_input = pd.DataFrame({
                "team": [row.home_team],
                "opponent": [row.away_team],
                "league": [row.league],
                "home": [1],
            })

            away_input = pd.DataFrame({
                "team": [row.away_team],
                "opponent": [row.home_team],
                "league": [row.league],
                "home": [0],
            })

            expected_home = model.predict(
                home_input
            )[0]

            expected_away = model.predict(
                away_input
            )[0]

            (
                prob_home,
                prob_draw,
                prob_away,
            ) = calculate_match_probabilities(
                expected_home,
                expected_away
            )

            predictions.append({
                "date": row.date,
                "league": row.league,
                "home_team": row.home_team,
                "away_team": row.away_team,

                "expected_home_goals":
                    expected_home,

                "expected_away_goals":
                    expected_away,

                "prob_home":
                    prob_home,

                "prob_draw":
                    prob_draw,

                "prob_away":
                    prob_away,

                "result":
                    row.result,
            })

        return pd.DataFrame(
            predictions
        )

    # --------------------
    # VALIDATION
    # --------------------

    validation_predictions = (
        predict_matches(validation)
    )

    probabilities = (
        validation_predictions[
            [
                "prob_away",
                "prob_draw",
                "prob_home",
            ]
        ].to_numpy()
    )

    validation_logloss = log_loss(
        validation["result"],
        probabilities,
        labels=["A", "D", "H"]
    )

    validation_predictions[
        "prediction"
    ] = (
        validation_predictions[
            [
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
        ]
        .idxmax(axis=1)
        .map({
            "prob_home": "H",
            "prob_draw": "D",
            "prob_away": "A",
        })
    )

    validation_accuracy = (
        validation_predictions[
            "prediction"
        ]
        == validation_predictions[
            "result"
        ]
    ).mean()

    print("\nPOISSON VALIDATION")
    print("------------------")

    print(
        f"Accuracy: {validation_accuracy:.3f}"
    )

    print(
        f"Log Loss: {validation_logloss:.3f}"
    )

    # --------------------
    # TEST
    # --------------------

    test_predictions = (
        predict_matches(test)
    )

    probabilities = (
        test_predictions[
            [
                "prob_away",
                "prob_draw",
                "prob_home",
            ]
        ].to_numpy()
    )

    test_logloss = log_loss(
        test["result"],
        probabilities,
        labels=["A", "D", "H"]
    )

    test_predictions[
        "prediction"
    ] = (
        test_predictions[
            [
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
        ]
        .idxmax(axis=1)
        .map({
            "prob_home": "H",
            "prob_draw": "D",
            "prob_away": "A",
        })
    )

    test_accuracy = (
        test_predictions[
            "prediction"
        ]
        == test_predictions[
            "result"
        ]
    ).mean()

    print("\nPOISSON TEST")
    print("------------------")

    print(
        f"Accuracy: {test_accuracy:.3f}"
    )

    print(
        f"Log Loss: {test_logloss:.3f}"
    )

    output = (
        PROCESSED_DIR
        / "poisson_predictions.csv"
    )

    test_predictions.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved -> {output}"
    )

    print("\nExample:\n")

    print(
        test_predictions[
            [
                "home_team",
                "away_team",
                "expected_home_goals",
                "expected_away_goals",
                "prob_home",
                "prob_draw",
                "prob_away",
                "result",
            ]
        ].tail(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run()
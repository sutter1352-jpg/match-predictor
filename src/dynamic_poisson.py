import pandas as pd
import numpy as np

from scipy.stats import poisson

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss

from .config import PROCESSED_DIR


MAX_GOALS = 10


NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "league"
]


ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def build_model():

    preprocessor = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES
        ),

        (
            "league",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        ),
    ])

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "poisson",
            PoissonRegressor(
                alpha=0.1,
                max_iter=2000
            )
        ),
    ])

    return model


def match_probabilities(
    expected_home,
    expected_away
):

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for hg in range(MAX_GOALS + 1):

        home_probability = poisson.pmf(
            hg,
            expected_home
        )

        for ag in range(MAX_GOALS + 1):

            away_probability = poisson.pmf(
                ag,
                expected_away
            )

            p = (
                home_probability
                * away_probability
            )

            if hg > ag:
                home_win += p

            elif hg == ag:
                draw += p

            else:
                away_win += p

    total = (
        home_win
        + draw
        + away_win
    )

    return (
        home_win / total,
        draw / total,
        away_win / total,
    )


def evaluate(
    matches,
    home_model,
    away_model,
):

    X = matches[ALL_FEATURES]

    expected_home = (
        home_model.predict(X)
    )

    expected_away = (
        away_model.predict(X)
    )

    predictions = []

    for i in range(len(matches)):

        (
            prob_home,
            prob_draw,
            prob_away,
        ) = match_probabilities(
            expected_home[i],
            expected_away[i],
        )

        predictions.append([
            prob_home,
            prob_draw,
            prob_away,
        ])

    probabilities = np.array(
        predictions
    )

    predicted_result = np.array(
        ["H", "D", "A"]
    )[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    actual = (
        matches["result"]
        .to_numpy()
    )

    accuracy = (
        predicted_result
        == actual
    ).mean()

    # Order required:
    # A, D, H

    logloss_probs = np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])

    score = log_loss(
        actual,
        logloss_probs,
        labels=["A", "D", "H"]
    )

    result = matches[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "result",
        ]
    ].copy()

    result[
        "expected_home_goals"
    ] = expected_home

    result[
        "expected_away_goals"
    ] = expected_away

    result[
        "prob_home"
    ] = probabilities[:, 0]

    result[
        "prob_draw"
    ] = probabilities[:, 1]

    result[
        "prob_away"
    ] = probabilities[:, 2]

    result[
        "prediction"
    ] = predicted_result

    return (
        accuracy,
        score,
        result,
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

    df = df.dropna(
        subset=
        ALL_FEATURES
        + [
            "home_goals",
            "away_goals",
            "result",
        ]
    ).copy()

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

    # -------------------------
    # HOME GOALS MODEL
    # -------------------------

    home_model = build_model()

    home_model.fit(
        train[ALL_FEATURES],
        train["home_goals"]
    )

    # -------------------------
    # AWAY GOALS MODEL
    # -------------------------

    away_model = build_model()

    away_model.fit(
        train[ALL_FEATURES],
        train["away_goals"]
    )

    # -------------------------
    # VALIDATION
    # -------------------------

    (
        validation_accuracy,
        validation_logloss,
        validation_predictions,
    ) = evaluate(
        validation,
        home_model,
        away_model,
    )

    print(
        "\nDYNAMIC POISSON VALIDATION"
    )

    print(
        "--------------------------"
    )

    print(
        f"Accuracy: {validation_accuracy:.3f}"
    )

    print(
        f"Log Loss: {validation_logloss:.3f}"
    )

    # -------------------------
    # TEST
    # -------------------------

    (
        test_accuracy,
        test_logloss,
        test_predictions,
    ) = evaluate(
        test,
        home_model,
        away_model,
    )

    print(
        "\nDYNAMIC POISSON TEST"
    )

    print(
        "--------------------------"
    )

    print(
        f"Accuracy: {test_accuracy:.3f}"
    )

    print(
        f"Log Loss: {test_logloss:.3f}"
    )

    output = (
        PROCESSED_DIR
        / "dynamic_poisson_predictions.csv"
    )

    test_predictions.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved -> {output}"
    )

    print(
        "\nExample predictions:\n"
    )

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
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
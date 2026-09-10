import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score

from .config import PROCESSED_DIR


# ==========================================
# FEATURES
# ==========================================

NUMERIC_FEATURES = [

    # Elo
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff_home_adjusted",
    "elo_expected_home_score",

    # Last 5
    "home_ppg_last5",
    "away_ppg_last5",

    "home_gfpg_last5",
    "away_gfpg_last5",

    "home_gapg_last5",
    "away_gapg_last5",

    "home_gdpg_last5",
    "away_gdpg_last5",

    # Last 10
    "home_ppg_last10",
    "away_ppg_last10",

    "home_gfpg_last10",
    "away_gfpg_last10",

    "home_gapg_last10",
    "away_gapg_last10",

    "home_gdpg_last10",
    "away_gdpg_last10",

    # Venue-specific
    "home_home_ppg_last5",
    "home_home_gfpg_last5",
    "home_home_gapg_last5",

    "away_away_ppg_last5",
    "away_away_gfpg_last5",
    "away_away_gapg_last5",

    # Exponentially weighted form
    "home_ewm_ppg",
    "away_ewm_ppg",

    "home_ewm_gfpg",
    "away_ewm_gfpg",

    "home_ewm_gapg",
    "away_ewm_gapg",

    # Rest
    "home_rest_days",
    "away_rest_days",

    # Available history
    "home_games_available",
    "away_games_available",
]


CATEGORICAL_FEATURES = [
    "league"
]


# ==========================================
# ADD DIFFERENCE FEATURES
# ==========================================

def add_difference_features(df):

    df = df.copy()

    df["ppg_diff_last5"] = (
        df["home_ppg_last5"]
        - df["away_ppg_last5"]
    )

    df["ppg_diff_last10"] = (
        df["home_ppg_last10"]
        - df["away_ppg_last10"]
    )

    df["gfpg_diff_last5"] = (
        df["home_gfpg_last5"]
        - df["away_gfpg_last5"]
    )

    df["gapg_diff_last5"] = (
        df["home_gapg_last5"]
        - df["away_gapg_last5"]
    )

    df["gdpg_diff_last5"] = (
        df["home_gdpg_last5"]
        - df["away_gdpg_last5"]
    )

    df["gdpg_diff_last10"] = (
        df["home_gdpg_last10"]
        - df["away_gdpg_last10"]
    )

    df["ewm_ppg_diff"] = (
        df["home_ewm_ppg"]
        - df["away_ewm_ppg"]
    )

    df["rest_diff"] = (
        df["home_rest_days"]
        - df["away_rest_days"]
    )

    return df


DIFFERENCE_FEATURES = [
    "ppg_diff_last5",
    "ppg_diff_last10",
    "gfpg_diff_last5",
    "gapg_diff_last5",
    "gdpg_diff_last5",
    "gdpg_diff_last10",
    "ewm_ppg_diff",
    "rest_diff",
]


ALL_NUMERIC = (
    NUMERIC_FEATURES
    + DIFFERENCE_FEATURES
)


# ==========================================
# BUILD MODEL
# ==========================================

def build_model():

    preprocessor = ColumnTransformer([

        (
            "numeric",
            "passthrough",
            ALL_NUMERIC
        ),

        (
            "league",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        ),
    ])

    classifier = HistGradientBoostingClassifier(

        learning_rate=0.04,

        max_iter=250,

        max_leaf_nodes=15,

        min_samples_leaf=40,

        l2_regularization=1.0,

        random_state=42
    )

    return Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        ),
    ])


# ==========================================
# PROBABILITY ORDER
# ==========================================

def probabilities_hda(
    model,
    X
):

    probabilities = (
        model.predict_proba(X)
    )

    classes = (
        model.named_steps[
            "classifier"
        ].classes_
    )

    index = {
        label: i
        for i, label in enumerate(classes)
    }

    # Always return:
    #
    # H, D, A

    return np.column_stack([

        probabilities[:, index["H"]],

        probabilities[:, index["D"]],

        probabilities[:, index["A"]],
    ])


# ==========================================
# BRIER SCORE
# ==========================================

def brier_score(
    actual,
    probabilities
):

    labels = [
        "H",
        "D",
        "A"
    ]

    score = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(labels):

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


# ==========================================
# EVALUATE
# ==========================================

def evaluate(
    name,
    actual,
    probabilities
):

    labels = np.array([
        "H",
        "D",
        "A"
    ])

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

    # sklearn expects:
    # A, D, H

    sklearn_probabilities = (
        np.column_stack([

            probabilities[:, 2],

            probabilities[:, 1],

            probabilities[:, 0],
        ])
    )

    loss = log_loss(
        actual,
        sklearn_probabilities,
        labels=[
            "A",
            "D",
            "H"
        ]
    )

    brier = brier_score(
        actual,
        probabilities
    )

    print(
        f"\n{name}"
    )

    print(
        "------------------------"
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


# ==========================================
# WALK FORWARD
# ==========================================

def walk_forward(
    df,
    start_date,
    end_date
):

    prediction_data = df[
        (
            df["date"]
            >= start_date
        )
        &
        (
            df["date"]
            < end_date
        )
    ].copy()

    prediction_data[
        "prediction_month"
    ] = (
        prediction_data["date"]
        .dt.to_period("M")
    )

    results = []

    months = sorted(
        prediction_data[
            "prediction_month"
        ].unique()
    )

    for month in months:

        current = prediction_data[
            prediction_data[
                "prediction_month"
            ] == month
        ].copy()

        first_date = (
            current["date"].min()
        )

        train = df[
            df["date"]
            < first_date
        ].copy()

        train = train.dropna(
            subset=["result"]
        )

        current = current.dropna(
            subset=["result"]
        )

        if len(current) == 0:
            continue

        print(
            f"\nPredicting {month}"
        )

        print(
            f"Training matches: {len(train)}"
        )

        print(
            f"Matches predicted: {len(current)}"
        )

        model = build_model()

        model.fit(
            train[
                ALL_NUMERIC
                + CATEGORICAL_FEATURES
            ],
            train["result"]
        )

        probabilities = (
            probabilities_hda(
                model,
                current[
                    ALL_NUMERIC
                    + CATEGORICAL_FEATURES
                ]
            )
        )

        for i, row in enumerate(
            current.itertuples()
        ):

            results.append({

                "date":
                    row.date,

                "league":
                    row.league,

                "home_team":
                    row.home_team,

                "away_team":
                    row.away_team,

                "result":
                    row.result,

                "prob_home":
                    probabilities[i, 0],

                "prob_draw":
                    probabilities[i, 1],

                "prob_away":
                    probabilities[i, 2],
            })

    return pd.DataFrame(
        results
    )


# ==========================================
# MAIN
# ==========================================

def run():

    path = (
        PROCESSED_DIR
        / "matches_advanced_features.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    df = add_difference_features(
        df
    )

    df = df.sort_values(
        "date"
    ).copy()

    print(
        f"Loaded {len(df):,} matches"
    )

    # ==================================
    # VALIDATION
    # 2024/25
    # ==================================

    validation_predictions = (
        walk_forward(

            df,

            pd.Timestamp(
                "2024-08-01"
            ),

            pd.Timestamp(
                "2025-08-01"
            )
        )
    )

    validation_probs = (
        validation_predictions[
            [
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
        ].to_numpy()
    )

    evaluate(
        "BOOSTING VALIDATION",
        validation_predictions[
            "result"
        ].to_numpy(),
        validation_probs
    )

    # ==================================
    # TEST
    # 2025/26
    # ==================================

    test_predictions = (
        walk_forward(

            df,

            pd.Timestamp(
                "2025-08-01"
            ),

            pd.Timestamp(
                "2026-08-01"
            )
        )
    )

    test_probs = (
        test_predictions[
            [
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
        ].to_numpy()
    )

    evaluate(
        "BOOSTING TEST",
        test_predictions["result"].to_numpy(),
        test_probs
    )

    # ==================================
    # SAVE VALIDATION + TEST
    # ==================================

    validation_output = (
        PROCESSED_DIR
        / "boosting_validation_predictions.csv"
    )

    test_output = (
        PROCESSED_DIR
        / "boosting_test_predictions.csv"
    )

    validation_predictions.to_csv(
        validation_output,
        index=False
    )

    test_predictions.to_csv(
        test_output,
        index=False
    )

    print(
        f"\nSaved validation -> {validation_output}"
    )

    print(
        f"Saved test -> {test_output}"
    )


if __name__ == "__main__":
    run()
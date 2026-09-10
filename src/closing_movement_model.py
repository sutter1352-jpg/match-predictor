import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss

from .config import PROCESSED_DIR


# ============================================================
# FEATURES AVAILABLE BEFORE THE MATCH
# ============================================================

FOOTBALL_FEATURES = [

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

    # Weighted recent form
    "home_ewm_ppg",
    "away_ewm_ppg",

    "home_ewm_gfpg",
    "away_ewm_gfpg",

    "home_ewm_gapg",
    "away_ewm_gapg",

    # Rest
    "home_rest_days",
    "away_rest_days",
]


# ============================================================
# ODDS -> FAIR PROBABILITIES
# ============================================================

def fair_probabilities(
    home_odds,
    draw_odds,
    away_odds
):

    raw = np.column_stack([
        1 / home_odds,
        1 / draw_odds,
        1 / away_odds,
    ])

    return (
        raw
        / raw.sum(
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# H/D/A PROBS -> TWO LOG RATIOS
#
# Draw is used as the reference category.
#
# z_home = log(P(Home) / P(Draw))
# z_away = log(P(Away) / P(Draw))
# ============================================================

def probabilities_to_logits(
    probabilities
):

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0
    )

    home_logit = np.log(
        probabilities[:, 0]
        / probabilities[:, 1]
    )

    away_logit = np.log(
        probabilities[:, 2]
        / probabilities[:, 1]
    )

    return (
        home_logit,
        away_logit
    )


# ============================================================
# LOG RATIOS -> H/D/A PROBABILITIES
# ============================================================

def logits_to_probabilities(
    home_logit,
    away_logit
):

    # Prevent numerical overflow.
    home_logit = np.clip(
        home_logit,
        -10,
        10
    )

    away_logit = np.clip(
        away_logit,
        -10,
        10
    )

    exp_home = np.exp(
        home_logit
    )

    exp_away = np.exp(
        away_logit
    )

    denominator = (
        1
        + exp_home
        + exp_away
    )

    home = (
        exp_home
        / denominator
    )

    draw = (
        1
        / denominator
    )

    away = (
        exp_away
        / denominator
    )

    return np.column_stack([
        home,
        draw,
        away,
    ])


# ============================================================
# LOG LOSS
# ============================================================

def calculate_logloss(
    actual,
    probabilities
):

    # sklearn expects A,D,H

    sklearn_probs = np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])

    return log_loss(
        actual,
        sklearn_probs,
        labels=[
            "A",
            "D",
            "H"
        ]
    )


# ============================================================
# BRIER
# ============================================================

def brier_score(
    actual,
    probabilities
):

    labels = [
        "H",
        "D",
        "A"
    ]

    total = 0.0

    for i, result in enumerate(actual):

        for j, label in enumerate(labels):

            observed = (
                1.0
                if result == label
                else 0.0
            )

            total += (
                probabilities[i, j]
                - observed
            ) ** 2

    return (
        total
        / len(actual)
    )


# ============================================================
# EVALUATE AGAINST RESULTS
# ============================================================

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

    predictions = labels[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    accuracy = (
        predictions
        == actual
    ).mean()

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
        "--------------------------"
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

    return loss


# ============================================================
# CREATE MARKET FEATURES
# ============================================================

def prepare_data(df):

    required_odds = [

        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    df = df.dropna(
        subset=required_odds
    ).copy()

    for column in required_odds:

        df = df[
            df[column] > 1
        ]

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------
    # Non-closing fair market probabilities
    # --------------------------------------------

    early_probs = fair_probabilities(

        df[
            "avg_home_odds"
        ].to_numpy(),

        df[
            "avg_draw_odds"
        ].to_numpy(),

        df[
            "avg_away_odds"
        ].to_numpy(),
    )

    # --------------------------------------------
    # Closing fair probabilities
    # --------------------------------------------

    closing_probs = fair_probabilities(

        df[
            "avg_close_home_odds"
        ].to_numpy(),

        df[
            "avg_close_draw_odds"
        ].to_numpy(),

        df[
            "avg_close_away_odds"
        ].to_numpy(),
    )

    df[
        "market_prob_home"
    ] = early_probs[:, 0]

    df[
        "market_prob_draw"
    ] = early_probs[:, 1]

    df[
        "market_prob_away"
    ] = early_probs[:, 2]

    # --------------------------------------------
    # Bookmaker overround
    # --------------------------------------------

    df[
        "market_overround"
    ] = (

        1 / df["avg_home_odds"]

        +

        1 / df["avg_draw_odds"]

        +

        1 / df["avg_away_odds"]

        - 1
    )

    # --------------------------------------------
    # Market log ratios
    # --------------------------------------------

    (
        early_home_logit,
        early_away_logit
    ) = probabilities_to_logits(
        early_probs
    )

    (
        closing_home_logit,
        closing_away_logit
    ) = probabilities_to_logits(
        closing_probs
    )

    df[
        "market_home_logit"
    ] = early_home_logit

    df[
        "market_away_logit"
    ] = early_away_logit

    # --------------------------------------------
    # TARGET:
    #
    # How much did the market move?
    # --------------------------------------------

    df[
        "target_home_move"
    ] = (
        closing_home_logit
        - early_home_logit
    )

    df[
        "target_away_move"
    ] = (
        closing_away_logit
        - early_away_logit
    )

    return (
        df,
        early_probs,
        closing_probs
    )


# ============================================================
# BUILD REGRESSOR
# ============================================================

def build_model():

    return HistGradientBoostingRegressor(

        learning_rate=0.03,

        max_iter=250,

        max_leaf_nodes=10,

        min_samples_leaf=60,

        l2_regularization=3.0,

        random_state=42
    )


# ============================================================
# MAIN
# ============================================================

def run():

    path = (
        PROCESSED_DIR
        / "matches_advanced_features.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    (
        df,
        all_early_probs,
        all_closing_probs
    ) = prepare_data(df)

    # ========================================================
    # DIFFERENCE FEATURES
    # ========================================================

    df[
        "ppg_diff_last5"
    ] = (
        df["home_ppg_last5"]
        - df["away_ppg_last5"]
    )

    df[
        "ppg_diff_last10"
    ] = (
        df["home_ppg_last10"]
        - df["away_ppg_last10"]
    )

    df[
        "gdpg_diff_last5"
    ] = (
        df["home_gdpg_last5"]
        - df["away_gdpg_last5"]
    )

    df[
        "gdpg_diff_last10"
    ] = (
        df["home_gdpg_last10"]
        - df["away_gdpg_last10"]
    )

    df[
        "rest_diff"
    ] = (
        df["home_rest_days"]
        - df["away_rest_days"]
    )

    extra_features = [

        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",

        "market_home_logit",
        "market_away_logit",

        "market_overround",

        "ppg_diff_last5",
        "ppg_diff_last10",

        "gdpg_diff_last5",
        "gdpg_diff_last10",

        "rest_diff",
    ]

    feature_columns = (
        FOOTBALL_FEATURES
        + extra_features
    )

    # ========================================================
    # SPLITS
    #
    # Training: before 2024/25
    # Validation: 2024/25
    # Test: 2025/26
    # ========================================================

    train = df[
        df["date"]
        < pd.Timestamp(
            "2024-08-01"
        )
    ].copy()

    validation = df[
        (
            df["date"]
            >= pd.Timestamp(
                "2024-08-01"
            )
        )
        &
        (
            df["date"]
            < pd.Timestamp(
                "2025-08-01"
            )
        )
    ].copy()

    test = df[
        df["date"]
        >= pd.Timestamp(
            "2025-08-01"
        )
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

    # ========================================================
    # TRAIN MOVEMENT MODELS
    # ========================================================

    home_model = build_model()

    away_model = build_model()

    home_model.fit(

        train[
            feature_columns
        ],

        train[
            "target_home_move"
        ]
    )

    away_model.fit(

        train[
            feature_columns
        ],

        train[
            "target_away_move"
        ]
    )

    # ========================================================
    # FUNCTION TO PREDICT CLOSING MARKET
    # ========================================================

    def predict_market(
        dataset,
        shrinkage
    ):

        predicted_home_move = (
            home_model.predict(
                dataset[
                    feature_columns
                ]
            )
        )

        predicted_away_move = (
            away_model.predict(
                dataset[
                    feature_columns
                ]
            )
        )

        predicted_home_logit = (

            dataset[
                "market_home_logit"
            ].to_numpy()

            +

            shrinkage
            * predicted_home_move
        )

        predicted_away_logit = (

            dataset[
                "market_away_logit"
            ].to_numpy()

            +

            shrinkage
            * predicted_away_move
        )

        return logits_to_probabilities(
            predicted_home_logit,
            predicted_away_logit
        )

    # ========================================================
    # FIND SHRINKAGE ON VALIDATION ONLY
    # ========================================================

    validation_actual = (
        validation["result"]
        .to_numpy()
    )

    best_shrinkage = 0.0
    best_loss = float("inf")

    for shrinkage in np.arange(
        0,
        1.51,
        0.05
    ):

        probabilities = (
            predict_market(
                validation,
                shrinkage
            )
        )

        loss = calculate_logloss(
            validation_actual,
            probabilities
        )

        if loss < best_loss:

            best_loss = loss
            best_shrinkage = (
                shrinkage
            )

    print(
        "\nBEST VALIDATION SHRINKAGE"
    )

    print(
        "--------------------------"
    )

    print(
        f"Shrinkage: "
        f"{best_shrinkage:.2f}"
    )

    print(
        f"Log Loss: "
        f"{best_loss:.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_early = validation[
        [
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
    ].to_numpy()

    validation_predicted = (
        predict_market(
            validation,
            best_shrinkage
        )
    )

    validation_closing = (
        fair_probabilities(

            validation[
                "avg_close_home_odds"
            ].to_numpy(),

            validation[
                "avg_close_draw_odds"
            ].to_numpy(),

            validation[
                "avg_close_away_odds"
            ].to_numpy(),
        )
    )

    evaluate(

        "NON-CLOSING VALIDATION",

        validation_actual,

        validation_early
    )

    evaluate(

        "PREDICTED CLOSING VALIDATION",

        validation_actual,

        validation_predicted
    )

    evaluate(

        "ACTUAL CLOSING VALIDATION",

        validation_actual,

        validation_closing
    )

    # ========================================================
    # TEST
    # ========================================================

    test_actual = (
        test["result"]
        .to_numpy()
    )

    test_early = test[
        [
            "market_prob_home",
            "market_prob_draw",
            "market_prob_away",
        ]
    ].to_numpy()

    test_predicted = (
        predict_market(
            test,
            best_shrinkage
        )
    )

    test_closing = (
        fair_probabilities(

            test[
                "avg_close_home_odds"
            ].to_numpy(),

            test[
                "avg_close_draw_odds"
            ].to_numpy(),

            test[
                "avg_close_away_odds"
            ].to_numpy(),
        )
    )

    evaluate(

        "NON-CLOSING TEST",

        test_actual,

        test_early
    )

    evaluate(

        "PREDICTED CLOSING TEST",

        test_actual,

        test_predicted
    )

    evaluate(

        "ACTUAL CLOSING TEST",

        test_actual,

        test_closing
    )

    # ========================================================
    # HOW CLOSE WERE WE TO ACTUAL CLOSING PRICES?
    # ========================================================

    probability_mae = np.mean(
        np.abs(
            test_predicted
            - test_closing
        )
    )

    baseline_mae = np.mean(
        np.abs(
            test_early
            - test_closing
        )
    )

    print(
        "\nCLOSING PRICE PREDICTION"
    )

    print(
        "--------------------------"
    )

    print(
        f"Non-closing -> Closing MAE: "
        f"{baseline_mae:.4f}"
    )

    print(
        f"Predicted -> Closing MAE:   "
        f"{probability_mae:.4f}"
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

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
        "market_home"
    ] = test_early[:, 0]

    output[
        "market_draw"
    ] = test_early[:, 1]

    output[
        "market_away"
    ] = test_early[:, 2]

    output[
        "predicted_close_home"
    ] = test_predicted[:, 0]

    output[
        "predicted_close_draw"
    ] = test_predicted[:, 1]

    output[
        "predicted_close_away"
    ] = test_predicted[:, 2]

    output[
        "actual_close_home"
    ] = test_closing[:, 0]

    output[
        "actual_close_draw"
    ] = test_closing[:, 1]

    output[
        "actual_close_away"
    ] = test_closing[:, 2]

    output[
        "predicted_home_move"
    ] = (
        test_predicted[:, 0]
        - test_early[:, 0]
    )

    output[
        "predicted_draw_move"
    ] = (
        test_predicted[:, 1]
        - test_early[:, 1]
    )

    output[
        "predicted_away_move"
    ] = (
        test_predicted[:, 2]
        - test_early[:, 2]
    )

    output_path = (
        PROCESSED_DIR
        / "closing_movement_predictions.csv"
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
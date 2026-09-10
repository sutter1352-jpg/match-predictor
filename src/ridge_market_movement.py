import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from .config import PROCESSED_DIR


# ============================================================
# BASIC HELPERS
# ============================================================

LABELS = ["H", "D", "A"]


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


def probabilities_to_logits(
    probabilities
):

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0
    )

    home = np.log(
        probabilities[:, 0]
        / probabilities[:, 1]
    )

    away = np.log(
        probabilities[:, 2]
        / probabilities[:, 1]
    )

    return home, away


def logits_to_probabilities(
    home_logit,
    away_logit
):

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

    return np.column_stack([
        exp_home / denominator,
        1 / denominator,
        exp_away / denominator,
    ])


def logloss(
    actual,
    probabilities
):

    # sklearn expects A, D, H

    ordered = np.column_stack([
        probabilities[:, 2],
        probabilities[:, 1],
        probabilities[:, 0],
    ])

    return log_loss(
        actual,
        ordered,
        labels=[
            "A",
            "D",
            "H"
        ]
    )


def brier(
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

    return (
        total
        / len(actual)
    )


def evaluate(
    name,
    actual,
    probabilities
):

    labels = np.array(
        LABELS
    )

    predicted = labels[
        np.argmax(
            probabilities,
            axis=1
        )
    ]

    accuracy = (
        predicted
        == actual
    ).mean()

    loss = logloss(
        actual,
        probabilities
    )

    score = brier(
        actual,
        probabilities
    )

    print(f"\n{name}")
    print("--------------------------")
    print(f"Matches:     {len(actual)}")
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Log Loss:    {loss:.4f}")
    print(f"Brier Score: {score:.4f}")

    return loss


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required = [

        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    df = (
        df
        .dropna(
            subset=required
        )
        .copy()
    )

    for column in required:

        df = df[
            df[column] > 1
        ]

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    early = fair_probabilities(

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

    closing = fair_probabilities(

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
        "market_home"
    ] = early[:, 0]

    df[
        "market_draw"
    ] = early[:, 1]

    df[
        "market_away"
    ] = early[:, 2]

    (
        market_home_logit,
        market_away_logit
    ) = probabilities_to_logits(
        early
    )

    (
        close_home_logit,
        close_away_logit
    ) = probabilities_to_logits(
        closing
    )

    df[
        "market_home_logit"
    ] = market_home_logit

    df[
        "market_away_logit"
    ] = market_away_logit

    df[
        "target_home_move"
    ] = (
        close_home_logit
        - market_home_logit
    )

    df[
        "target_away_move"
    ] = (
        close_away_logit
        - market_away_logit
    )

    # ========================================================
    # DIFFERENCE FEATURES
    # ========================================================

    df[
        "elo_diff"
    ] = (
        df[
            "elo_diff_home_adjusted"
        ]
    )

    df[
        "ppg5_diff"
    ] = (
        df["home_ppg_last5"]
        - df["away_ppg_last5"]
    )

    df[
        "ppg10_diff"
    ] = (
        df["home_ppg_last10"]
        - df["away_ppg_last10"]
    )

    df[
        "gf5_diff"
    ] = (
        df["home_gfpg_last5"]
        - df["away_gfpg_last5"]
    )

    df[
        "ga5_diff"
    ] = (
        df["home_gapg_last5"]
        - df["away_gapg_last5"]
    )

    df[
        "gd5_diff"
    ] = (
        df["home_gdpg_last5"]
        - df["away_gdpg_last5"]
    )

    df[
        "gd10_diff"
    ] = (
        df["home_gdpg_last10"]
        - df["away_gdpg_last10"]
    )

    df[
        "ewm_ppg_diff"
    ] = (
        df["home_ewm_ppg"]
        - df["away_ewm_ppg"]
    )

    df[
        "rest_diff"
    ] = (
        df["home_rest_days"]
        - df["away_rest_days"]
    )

    return df, early, closing


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Existing market
    "market_home",
    "market_draw",
    "market_away",

    "market_home_logit",
    "market_away_logit",

    # Football information
    "elo_diff",

    "ppg5_diff",
    "ppg10_diff",

    "gf5_diff",
    "ga5_diff",

    "gd5_diff",
    "gd10_diff",

    "ewm_ppg_diff",

    "rest_diff",
]


# ============================================================
# MODEL
# ============================================================

def build_model(alpha):

    return Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ridge",
            Ridge(
                alpha=alpha
            )
        )
    ])


# ============================================================
# MAIN
# ============================================================

def run():

    df = pd.read_csv(

        PROCESSED_DIR
        / "matches_advanced_features.csv",

        parse_dates=["date"]
    )

    (
        df,
        early_probs,
        closing_probs
    ) = prepare_data(df)

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
    # SEARCH REGULARIZATION USING VALIDATION ONLY
    # ========================================================

    alphas = [
        0.01,
        0.1,
        1,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
    ]

    best_alpha = None
    best_loss = float(
        "inf"
    )

    best_home_model = None
    best_away_model = None

    for alpha in alphas:

        home_model = build_model(
            alpha
        )

        away_model = build_model(
            alpha
        )

        home_model.fit(

            train[FEATURES],

            train[
                "target_home_move"
            ]
        )

        away_model.fit(

            train[FEATURES],

            train[
                "target_away_move"
            ]
        )

        home_move = (
            home_model.predict(
                validation[FEATURES]
            )
        )

        away_move = (
            away_model.predict(
                validation[FEATURES]
            )
        )

        predicted = logits_to_probabilities(

            validation[
                "market_home_logit"
            ].to_numpy()
            + home_move,

            validation[
                "market_away_logit"
            ].to_numpy()
            + away_move
        )

        loss = logloss(

            validation[
                "result"
            ].to_numpy(),

            predicted
        )

        print(
            f"Alpha {alpha:<7} "
            f"Validation Log Loss: "
            f"{loss:.4f}"
        )

        if loss < best_loss:

            best_loss = loss
            best_alpha = alpha

            best_home_model = (
                home_model
            )

            best_away_model = (
                away_model
            )

    print(
        "\nBEST RIDGE MODEL"
    )

    print(
        "--------------------------"
    )

    print(
        f"Alpha: {best_alpha}"
    )

    print(
        f"Validation Log Loss: "
        f"{best_loss:.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_market = validation[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()

    validation_home_move = (
        best_home_model.predict(
            validation[FEATURES]
        )
    )

    validation_away_move = (
        best_away_model.predict(
            validation[FEATURES]
        )
    )

    validation_predicted = (
        logits_to_probabilities(

            validation[
                "market_home_logit"
            ].to_numpy()
            + validation_home_move,

            validation[
                "market_away_logit"
            ].to_numpy()
            + validation_away_move
        )
    )

    evaluate(

        "MARKET VALIDATION",

        validation[
            "result"
        ].to_numpy(),

        validation_market
    )

    evaluate(

        "RIDGE VALIDATION",

        validation[
            "result"
        ].to_numpy(),

        validation_predicted
    )

    # ========================================================
    # TEST
    # ========================================================

    test_market = test[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()

    test_home_move = (
        best_home_model.predict(
            test[FEATURES]
        )
    )

    test_away_move = (
        best_away_model.predict(
            test[FEATURES]
        )
    )

    test_predicted = (
        logits_to_probabilities(

            test[
                "market_home_logit"
            ].to_numpy()
            + test_home_move,

            test[
                "market_away_logit"
            ].to_numpy()
            + test_away_move
        )
    )

    test_closing = fair_probabilities(

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

    evaluate(

        "NON-CLOSING TEST",

        test[
            "result"
        ].to_numpy(),

        test_market
    )

    evaluate(

        "RIDGE PREDICTED TEST",

        test[
            "result"
        ].to_numpy(),

        test_predicted
    )

    evaluate(

        "ACTUAL CLOSING TEST",

        test[
            "result"
        ].to_numpy(),

        test_closing
    )

    # ========================================================
    # CLOSING PRICE MAE
    # ========================================================

    baseline_mae = (
        np.mean(
            np.abs(
                test_market
                - test_closing
            )
        )
    )

    ridge_mae = (
        np.mean(
            np.abs(
                test_predicted
                - test_closing
            )
        )
    )

    print(
        "\nCLOSING PRICE MAE"
    )

    print(
        "--------------------------"
    )

    print(
        f"Market -> Closing: "
        f"{baseline_mae:.4f}"
    )

    print(
        f"Ridge -> Closing:  "
        f"{ridge_mae:.4f}"
    )

    # ========================================================
    # DIRECTION ACCURACY
    # ========================================================

    actual_move = (
        test_closing
        - test_market
    )

    predicted_move = (
        test_predicted
        - test_market
    )

    mask = (
        np.abs(actual_move)
        >= 0.005
    )

    direction_correct = (

        np.sign(
            actual_move[mask]
        )

        ==

        np.sign(
            predicted_move[mask]
        )
    )

    print(
        "\nMOVEMENT DIRECTION"
    )

    print(
        "--------------------------"
    )

    print(
        "Outcomes with actual "
        "move >= 0.5%:",
        len(direction_correct)
    )

    print(
        "Direction accuracy:",
        f"{direction_correct.mean():.3f}"
    )

    # ========================================================
    # SAVE
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
    ] = test_market[:, 0]

    output[
        "market_draw"
    ] = test_market[:, 1]

    output[
        "market_away"
    ] = test_market[:, 2]

    output[
        "ridge_home"
    ] = test_predicted[:, 0]

    output[
        "ridge_draw"
    ] = test_predicted[:, 1]

    output[
        "ridge_away"
    ] = test_predicted[:, 2]

    output[
        "close_home"
    ] = test_closing[:, 0]

    output[
        "close_draw"
    ] = test_closing[:, 1]

    output[
        "close_away"
    ] = test_closing[:, 2]

    output.to_csv(

        PROCESSED_DIR
        / "ridge_market_predictions.csv",

        index=False
    )


if __name__ == "__main__":
    run()
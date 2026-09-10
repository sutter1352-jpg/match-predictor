import numpy as np
import pandas as pd

from sklearn.metrics import log_loss

from .config import PROCESSED_DIR

from .ridge_market_movement import (
    prepare_data,
    FEATURES,
    build_model,
    logits_to_probabilities,
    fair_probabilities,
)


ALPHAS = [
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

THRESHOLDS = [
    0.0025,
    0.0050,
]

SEASONS = [

    # target season,
    # validation start,
    # target start,
    # target end

    (
        "2022/23",
        "2021-08-01",
        "2022-08-01",
        "2023-08-01",
    ),

    (
        "2023/24",
        "2022-08-01",
        "2023-08-01",
        "2024-08-01",
    ),

    (
        "2024/25",
        "2023-08-01",
        "2024-08-01",
        "2025-08-01",
    ),

    (
        "2025/26",
        "2024-08-01",
        "2025-08-01",
        "2026-08-01",
    ),
]


def sklearn_probs(probabilities):

    # H,D,A -> A,D,H

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
        sklearn_probs(probabilities),
        labels=["A", "D", "H"]
    )


def get_closing_probs(df):

    return fair_probabilities(

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


def market_probs(df):

    return df[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()


def predict(
    df,
    home_model,
    away_model
):

    home_move = home_model.predict(
        df[FEATURES]
    )

    away_move = away_model.predict(
        df[FEATURES]
    )

    return logits_to_probabilities(

        df[
            "market_home_logit"
        ].to_numpy()
        + home_move,

        df[
            "market_away_logit"
        ].to_numpy()
        + away_move
    )


def choose_alpha(
    train,
    validation
):

    best_alpha = None
    best_mae = float("inf")

    closing = get_closing_probs(
        validation
    )

    for alpha in ALPHAS:

        home_model = build_model(
            alpha
        )

        away_model = build_model(
            alpha
        )

        home_model.fit(
            train[FEATURES],
            train["target_home_move"]
        )

        away_model.fit(
            train[FEATURES],
            train["target_away_move"]
        )

        predicted = predict(
            validation,
            home_model,
            away_model
        )

        mae = np.mean(
            np.abs(
                predicted
                - closing
            )
        )

        if mae < best_mae:

            best_mae = mae
            best_alpha = alpha

    return (
        best_alpha,
        best_mae
    )


def create_signals(
    matches,
    predicted
):

    market = market_probs(
        matches
    )

    closing = get_closing_probs(
        matches
    )

    predicted_moves = (
        predicted
        - market
    )

    actual_moves = (
        closing
        - market
    )

    outcome_names = np.array([
        "home",
        "draw",
        "away",
    ])

    strongest_index = np.argmax(
        np.abs(predicted_moves),
        axis=1
    )

    rows = []

    for i, row in enumerate(
        matches.itertuples()
    ):

        j = strongest_index[i]

        predicted_move = (
            predicted_moves[i, j]
        )

        actual_move = (
            actual_moves[i, j]
        )

        rows.append({

            "date":
                row.date,

            "league":
                row.league,

            "home_team":
                row.home_team,

            "away_team":
                row.away_team,

            "outcome":
                outcome_names[j],

            "predicted_move":
                predicted_move,

            "predicted_move_abs":
                abs(predicted_move),

            "actual_move":
                actual_move,

            # Positive = market later moved
            # in our predicted direction.
            "directional_clv":
                np.sign(
                    predicted_move
                )
                * actual_move,

            "correct_direction":
                np.sign(
                    predicted_move
                )
                ==
                np.sign(
                    actual_move
                ),
        })

    return pd.DataFrame(
        rows
    )


def run():

    df = pd.read_csv(

        PROCESSED_DIR
        / "matches_advanced_features.csv",

        parse_dates=["date"]
    )

    df, _, _ = prepare_data(
        df
    )

    all_signals = []

    print(
        "\nROLLING OUT-OF-SAMPLE TEST"
    )

    print(
        "=" * 70
    )

    for (
        season,
        validation_start,
        test_start,
        test_end,
    ) in SEASONS:

        validation_start = (
            pd.Timestamp(
                validation_start
            )
        )

        test_start = (
            pd.Timestamp(
                test_start
            )
        )

        test_end = (
            pd.Timestamp(
                test_end
            )
        )

        # ==================================
        # TRAINING
        # ==================================

        train = df[
            df["date"]
            < validation_start
        ].copy()

        # ==================================
        # VALIDATION
        # Used only to select alpha.
        # ==================================

        validation = df[
            (
                df["date"]
                >= validation_start
            )
            &
            (
                df["date"]
                < test_start
            )
        ].copy()

        # ==================================
        # TRUE OUT-OF-SAMPLE TARGET SEASON
        # ==================================

        test = df[
            (
                df["date"]
                >= test_start
            )
            &
            (
                df["date"]
                < test_end
            )
        ].copy()

        if (
            len(train) == 0
            or len(validation) == 0
            or len(test) == 0
        ):
            continue

        # ==================================
        # CHOOSE ALPHA WITHOUT TEST DATA
        # ==================================

        (
            best_alpha,
            validation_mae
        ) = choose_alpha(
            train,
            validation
        )

        # ==================================
        # RETRAIN USING EVERYTHING AVAILABLE
        # BEFORE THE TARGET SEASON
        # ==================================

        final_train = df[
            df["date"]
            < test_start
        ].copy()

        home_model = build_model(
            best_alpha
        )

        away_model = build_model(
            best_alpha
        )

        home_model.fit(

            final_train[FEATURES],

            final_train[
                "target_home_move"
            ]
        )

        away_model.fit(

            final_train[FEATURES],

            final_train[
                "target_away_move"
            ]
        )

        predicted = predict(
            test,
            home_model,
            away_model
        )

        market = market_probs(
            test
        )

        closing = get_closing_probs(
            test
        )

        actual = (
            test["result"]
            .to_numpy()
        )

        market_loss = (
            calculate_logloss(
                actual,
                market
            )
        )

        model_loss = (
            calculate_logloss(
                actual,
                predicted
            )
        )

        market_close_mae = np.mean(
            np.abs(
                market
                - closing
            )
        )

        predicted_close_mae = np.mean(
            np.abs(
                predicted
                - closing
            )
        )

        signals = create_signals(
            test,
            predicted
        )

        signals[
            "season"
        ] = season

        all_signals.append(
            signals
        )

        print(
            f"\n{season}"
        )

        print(
            "-" * 70
        )

        print(
            f"Matches: {len(test)}"
        )

        print(
            f"Selected alpha: "
            f"{best_alpha}"
        )

        print(
            f"Validation closing MAE: "
            f"{validation_mae:.4f}"
        )

        print(
            f"Market Log Loss: "
            f"{market_loss:.4f}"
        )

        print(
            f"Ridge Log Loss:  "
            f"{model_loss:.4f}"
        )

        print(
            f"Market -> Close MAE: "
            f"{market_close_mae:.4f}"
        )

        print(
            f"Ridge -> Close MAE:  "
            f"{predicted_close_mae:.4f}"
        )

        for threshold in THRESHOLDS:

            subset = signals[
                signals[
                    "predicted_move_abs"
                ]
                >= threshold
            ]

            if len(subset) == 0:
                continue

            print(
                f"\nSignal >= "
                f"{threshold:.2%}"
            )

            print(
                f"Signals: "
                f"{len(subset)}"
            )

            print(
                f"Direction: "
                f"{subset['correct_direction'].mean():.2%}"
            )

            print(
                f"Average CLV: "
                f"{subset['directional_clv'].mean():.3%}"
            )

    # ==========================================
    # COMBINE ALL TRUE OUT-OF-SAMPLE SEASONS
    # ==========================================

    all_signals = pd.concat(
        all_signals,
        ignore_index=True
    )

    print(
        "\n\nPOOLED OUT-OF-SAMPLE RESULTS"
    )

    print(
        "=" * 70
    )

    for threshold in THRESHOLDS:

        subset = all_signals[
            all_signals[
                "predicted_move_abs"
            ]
            >= threshold
        ]

        print(
            f"\nThreshold >= "
            f"{threshold:.2%}"
        )

        print(
            f"Signals: "
            f"{len(subset)}"
        )

        print(
            f"Direction accuracy: "
            f"{subset['correct_direction'].mean():.2%}"
        )

        print(
            f"Average CLV: "
            f"{subset['directional_clv'].mean():.3%}"
        )

        print(
            f"Median CLV: "
            f"{subset['directional_clv'].median():.3%}"
        )

    # ==========================================
    # 0.50% SIGNAL BY SEASON
    # ==========================================

    strong = all_signals[
        all_signals[
            "predicted_move_abs"
        ]
        >= 0.005
    ]

    summary = (
        strong
        .groupby("season")
        .agg(

            signals=(
                "correct_direction",
                "size"
            ),

            direction_accuracy=(
                "correct_direction",
                "mean"
            ),

            average_clv=(
                "directional_clv",
                "mean"
            ),
        )
    )

    print(
        "\n\n0.50% SIGNAL BY SEASON"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            formatters={

                "direction_accuracy":
                    lambda x:
                    f"{x:.2%}",

                "average_clv":
                    lambda x:
                    f"{x:.3%}",
            }
        )
    )

    all_signals.to_csv(

        PROCESSED_DIR
        / "rolling_oos_signals.csv",

        index=False
    )


if __name__ == "__main__":
    run()
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR

from .ridge_market_movement import (
    prepare_data,
    FEATURES,
    build_model,
    logits_to_probabilities,
    fair_probabilities,
)


# ============================================================
# SETTINGS
# ============================================================

STAKE = 10.00

MIN_SIGNAL = 0.0025      # +0.25 percentage points
STRONG_SIGNAL = 0.0050   # +0.50 percentage points

MIN_EV = 0.01            # minimum +1% expected ROI


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


SEASONS = [

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


# ============================================================
# CLOSING PROBABILITIES
# ============================================================

def closing_probabilities(df):

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


# ============================================================
# PREDICT
# ============================================================

def predict_probabilities(
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


# ============================================================
# SELECT ALPHA
#
# Uses only the season BEFORE the test season.
# ============================================================

def choose_alpha(
    train,
    validation
):

    closing = closing_probabilities(
        validation
    )

    best_alpha = None
    best_mae = float("inf")

    for alpha in ALPHAS:

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

        predicted = predict_probabilities(
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

    return best_alpha


# ============================================================
# EXECUTION ODDS
#
# Prefer Max odds.
# Fall back to Bet365 if unavailable.
# ============================================================

def execution_odds(row):

    max_odds = np.array([

        row.get(
            "max_home_odds",
            np.nan
        ),

        row.get(
            "max_draw_odds",
            np.nan
        ),

        row.get(
            "max_away_odds",
            np.nan
        ),
    ])

    if (
        np.all(np.isfinite(max_odds))
        and
        np.all(max_odds > 1)
    ):

        return (
            max_odds,
            "Max market"
        )

    b365 = np.array([

        row.get(
            "b365_home_odds",
            np.nan
        ),

        row.get(
            "b365_draw_odds",
            np.nan
        ),

        row.get(
            "b365_away_odds",
            np.nan
        ),
    ])

    if (
        np.all(np.isfinite(b365))
        and
        np.all(b365 > 1)
    ):

        return (
            b365,
            "Bet365"
        )

    return None, None


# ============================================================
# CREATE PAPER BETS
# ============================================================

def create_bets(
    matches,
    model_probs
):

    market_probs = matches[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()

    close_probs = (
        closing_probabilities(
            matches
        )
    )

    result_labels = np.array([
        "H",
        "D",
        "A",
    ])

    bets = []

    for i, (_, row) in enumerate(
        matches.iterrows()
    ):

        odds, odds_source = (
            execution_odds(row)
        )

        if odds is None:
            continue

        best_candidate = None

        for j in range(3):

            model_probability = (
                model_probs[i, j]
            )

            market_probability = (
                market_probs[i, j]
            )

            signal = (
                model_probability
                - market_probability
            )

            break_even_probability = (
                1 / odds[j]
            )

            expected_roi = (
                model_probability
                * odds[j]
                - 1
            )

            # --------------------------------
            # BET FILTER
            # --------------------------------

            if signal < MIN_SIGNAL:
                continue

            if expected_roi < MIN_EV:
                continue

            if j == 0:

                bet_name = (
                    row["home_team"]
                )

            elif j == 1:

                bet_name = "DRAW"

            else:

                bet_name = (
                    row["away_team"]
                )

            actual_result = (
                row["result"]
            )

            won = (
                actual_result
                == result_labels[j]
            )

            if won:

                profit = (
                    STAKE
                    * (
                        odds[j]
                        - 1
                    )
                )

            else:

                profit = -STAKE

            actual_clv = (
                close_probs[i, j]
                - market_probability
            )

            candidate = {

                "date":
                    row["date"],

                "season":
                    row.get(
                        "simulation_season",
                        ""
                    ),

                "league":
                    row["league"],

                "home_team":
                    row["home_team"],

                "away_team":
                    row["away_team"],

                "bet":
                    bet_name,

                "bet_result":
                    result_labels[j],

                "odds":
                    odds[j],

                "odds_source":
                    odds_source,

                "model_probability":
                    model_probability,

                "market_probability":
                    market_probability,

                "break_even_probability":
                    break_even_probability,

                "signal":
                    signal,

                "expected_roi":
                    expected_roi,

                "actual_clv":
                    actual_clv,

                "actual_result":
                    actual_result,

                "won":
                    won,

                "stake":
                    STAKE,

                "profit":
                    profit,

                "signal_label":
                    (
                        "STRONG"
                        if signal
                        >= STRONG_SIGNAL
                        else
                        "STANDARD"
                    ),
            }

            # --------------------------------
            # Maximum ONE bet per match.
            #
            # Pick the highest expected ROI.
            # --------------------------------

            if (
                best_candidate is None
                or
                candidate[
                    "expected_roi"
                ]
                >
                best_candidate[
                    "expected_roi"
                ]
            ):

                best_candidate = (
                    candidate
                )

        if best_candidate is not None:

            bets.append(
                best_candidate
            )

    return pd.DataFrame(
        bets
    )


# ============================================================
# HISTORICAL ROLLING SIMULATION
# ============================================================

def historical_simulation(df):

    all_bets = []

    print(
        "\n"
        + "=" * 95
    )

    print(
        "HISTORICAL OUT-OF-SAMPLE "
        "PAPER BETTING"
    )

    print(
        "=" * 95
    )

    for (
        season,
        validation_start,
        test_start,
        test_end,
    ) in SEASONS:

        validation_start = pd.Timestamp(
            validation_start
        )

        test_start = pd.Timestamp(
            test_start
        )

        test_end = pd.Timestamp(
            test_end
        )

        train = df[
            df["date"]
            < validation_start
        ].copy()

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
            or
            len(validation) == 0
            or
            len(test) == 0
        ):

            continue

        alpha = choose_alpha(
            train,
            validation
        )

        final_train = df[
            df["date"]
            < test_start
        ].copy()

        home_model = build_model(
            alpha
        )

        away_model = build_model(
            alpha
        )

        home_model.fit(

            final_train[
                FEATURES
            ],

            final_train[
                "target_home_move"
            ]
        )

        away_model.fit(

            final_train[
                FEATURES
            ],

            final_train[
                "target_away_move"
            ]
        )

        predicted = (
            predict_probabilities(
                test,
                home_model,
                away_model
            )
        )

        test = test.copy()

        test[
            "simulation_season"
        ] = season

        bets = create_bets(
            test,
            predicted
        )

        if len(bets) == 0:

            print(
                f"\n{season}: "
                f"No qualifying bets."
            )

            continue

        all_bets.append(
            bets
        )

        summarize_bets(
            bets,
            season
        )

    if not all_bets:

        return pd.DataFrame()

    return pd.concat(
        all_bets,
        ignore_index=True
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_bets(
    bets,
    title
):

    if len(bets) == 0:

        print(
            f"\n{title}"
        )

        print(
            "No qualifying bets."
        )

        return

    total_bets = len(
        bets
    )

    wins = int(
        bets["won"].sum()
    )

    losses = (
        total_bets - wins
    )

    wagered = (
        bets["stake"].sum()
    )

    profit = (
        bets["profit"].sum()
    )

    roi = (
        profit / wagered
        if wagered > 0
        else 0
    )

    average_ev = (
        bets[
            "expected_roi"
        ].mean()
    )

    average_clv = (
        bets[
            "actual_clv"
        ].mean()
    )

    print(
        f"\n{title}"
    )

    print(
        "-" * 55
    )

    print(
        f"Bets:        {total_bets}"
    )

    print(
        f"Wins:        {wins}"
    )

    print(
        f"Losses:      {losses}"
    )

    print(
        f"Win rate:    "
        f"{wins / total_bets:.2%}"
    )

    print(
        f"Wagered:     "
        f"${wagered:,.2f}"
    )

    print(
        f"Profit/Loss: "
        f"${profit:+,.2f}"
    )

    print(
        f"ROI:         "
        f"{roi:+.2%}"
    )

    print(
        f"Average EV:  "
        f"{average_ev:+.2%}"
    )

    print(
        f"Average CLV: "
        f"{average_clv:+.2%}"
    )


# ============================================================
# CLEAR BET TABLE
# ============================================================

def print_bet_table(
    bets,
    title
):

    print(
        "\n"
        + "=" * 120
    )

    print(title)

    print(
        "=" * 120
    )

    if len(bets) == 0:

        print(
            "NO QUALIFYING PAPER BETS"
        )

        return

    display = bets.copy()

    display["match"] = (
        display["home_team"]
        + " vs "
        + display["away_team"]
    )

    display["model"] = (
        display[
            "model_probability"
        ]
        * 100
    )

    display["break_even"] = (
        display[
            "break_even_probability"
        ]
        * 100
    )

    display["edge"] = (

        display[
            "model_probability"
        ]

        -

        display[
            "break_even_probability"
        ]

    ) * 100

    display["ev"] = (
        display[
            "expected_roi"
        ]
        * 100
    )

    display["clv"] = (
        display[
            "actual_clv"
        ]
        * 100
    )

    display["outcome"] = np.where(
        display["won"],
        "WIN",
        "LOSS"
    )

    display["pnl"] = (
        display["profit"]
    )

    display = (
        display
        .sort_values(
            [
                "signal_label",
                "expected_roi",
            ],
            ascending=[
                True,
                False,
            ]
        )
    )

    columns = [

        "date",
        "league",
        "match",
        "bet",
        "signal_label",
        "odds",
        "model",
        "break_even",
        "edge",
        "ev",
        "outcome",
        "pnl",
        "clv",
    ]

    print(

        display[
            columns
        ].to_string(

            index=False,

            formatters={

                "odds":
                    lambda x:
                    f"{x:.2f}",

                "model":
                    lambda x:
                    f"{x:.1f}%",

                "break_even":
                    lambda x:
                    f"{x:.1f}%",

                "edge":
                    lambda x:
                    f"{x:+.1f}%",

                "ev":
                    lambda x:
                    f"{x:+.1f}%",

                "pnl":
                    lambda x:
                    f"${x:+.2f}",

                "clv":
                    lambda x:
                    f"{x:+.2f}%",
            }
        )
    )


# ============================================================
# LAST WEEK
# ============================================================

def last_week_simulation(
    full_matches
):

    path = (
        PROCESSED_DIR
        / "last_week_predictions.csv"
    )

    if not path.exists():

        print(
            "\nCould not find "
            "last_week_predictions.csv"
        )

        print(
            "Run:"
        )

        print(
            "python -m "
            "src.last_week_simulation"
        )

        return pd.DataFrame()

    predictions = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    merge_columns = [
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
    ]

    odds_columns = [

        "max_home_odds",
        "max_draw_odds",
        "max_away_odds",

        "b365_home_odds",
        "b365_draw_odds",
        "b365_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    available = [
        c
        for c in odds_columns
        if c in full_matches.columns
    ]

    merged = predictions.merge(

        full_matches[
            merge_columns
            + available
        ],

        on=merge_columns,

        how="left"
    )

    # ---------------------------------------------
    # Give create_bets() the column names it expects.
    # ---------------------------------------------

    merged[
        "market_home"
    ] = merged[
        "market_home"
    ]

    merged[
        "market_draw"
    ] = merged[
        "market_draw"
    ]

    merged[
        "market_away"
    ] = merged[
        "market_away"
    ]

    # Closing odds already merged from main dataset.

    model_probs = merged[
        [
            "model_home",
            "model_draw",
            "model_away",
        ]
    ].to_numpy()

    bets = create_bets(
        merged,
        model_probs
    )

    return bets


# ============================================================
# MAIN
# ============================================================

def run():

    print(
        "\nPAPER BET RULE"
    )

    print(
        "=" * 60
    )

    print(
        "Minimum predicted market move: "
        f"{MIN_SIGNAL:.2%}"
    )

    print(
        "Strong signal threshold:       "
        f"{STRONG_SIGNAL:.2%}"
    )

    print(
        "Minimum expected ROI:          "
        f"{MIN_EV:.2%}"
    )

    print(
        "Stake per bet:                 "
        f"${STAKE:.2f}"
    )

    print(
        "Maximum bets per match:        1"
    )

    # ========================================================
    # LOAD FULL DATASET
    # ========================================================

    raw = pd.read_csv(

        PROCESSED_DIR
        / "matches_advanced_features.csv",

        parse_dates=["date"]
    )

    full_matches = raw.copy()

    df, _, _ = prepare_data(
        raw
    )

    # ========================================================
    # HISTORICAL OOS
    # ========================================================

    historical_bets = (
        historical_simulation(
            df
        )
    )

    if len(historical_bets) > 0:

        summarize_bets(
            historical_bets,
            "ALL HISTORICAL OOS BETS"
        )

        strong_historical = (
            historical_bets[
                historical_bets[
                    "signal"
                ]
                >= STRONG_SIGNAL
            ]
        )

        summarize_bets(
            strong_historical,
            "STRONG HISTORICAL BETS"
        )

    # ========================================================
    # LAST WEEK
    # ========================================================

    recent_bets = (
        last_week_simulation(
            full_matches
        )
    )

    print_bet_table(
        recent_bets,
        "LAST WEEK — MODEL PAPER BETS"
    )

    summarize_bets(
        recent_bets,
        "LAST WEEK SUMMARY"
    )

    if len(recent_bets) > 0:

        strong_recent = recent_bets[
            recent_bets[
                "signal"
            ]
            >= STRONG_SIGNAL
        ]

        print_bet_table(
            strong_recent,
            "LAST WEEK — STRONG SIGNALS ONLY"
        )

        summarize_bets(
            strong_recent,
            "LAST WEEK STRONG SIGNAL SUMMARY"
        )

    # ========================================================
    # SAVE
    # ========================================================

    if len(historical_bets) > 0:

        historical_bets.to_csv(

            PROCESSED_DIR
            / "historical_paper_bets.csv",

            index=False
        )

    if len(recent_bets) > 0:

        recent_bets.to_csv(

            PROCESSED_DIR
            / "last_week_paper_bets.csv",

            index=False
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 60
    )

    print(
        "data/processed/"
        "historical_paper_bets.csv"
    )

    print(
        "data/processed/"
        "last_week_paper_bets.csv"
    )


if __name__ == "__main__":
    run()
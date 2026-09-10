import numpy as np
import pandas as pd

from .config import PROCESSED_DIR

from .ridge_market_movement import (
    prepare_data,
    FEATURES,
    build_model,
    fair_probabilities,
)

from .paper_betting_simulation import (
    SEASONS,
    choose_alpha,
    predict_probabilities,
)


STAKE = 10.00

# Model must move probability at least +0.25 percentage points
MIN_SIGNAL = 0.0025

# Model must estimate at least +1% expected ROI
MIN_EV = 0.01


OUTCOME_CODES = ["H", "D", "A"]


def closing_probs(df):

    return fair_probabilities(
        df["avg_close_home_odds"].to_numpy(),
        df["avg_close_draw_odds"].to_numpy(),
        df["avg_close_away_odds"].to_numpy(),
    )


def create_candidates(matches, model_probs):

    market_probs = matches[
        [
            "market_home",
            "market_draw",
            "market_away",
        ]
    ].to_numpy()

    close = closing_probs(matches)

    candidates = []

    for i, (_, row) in enumerate(matches.iterrows()):

        # -----------------------------------------
        # CONSERVATIVE EXECUTION:
        # Bet365 only
        # -----------------------------------------

        odds = np.array([
            row.get("b365_home_odds", np.nan),
            row.get("b365_draw_odds", np.nan),
            row.get("b365_away_odds", np.nan),
        ])

        if (
            not np.all(np.isfinite(odds))
            or
            np.any(odds <= 1)
        ):
            continue

        best = None

        for j in range(3):

            model_probability = model_probs[i, j]

            market_probability = market_probs[i, j]

            # Ridge adjustment versus fair market
            signal = (
                model_probability
                - market_probability
            )

            # Only consider outcomes where
            # our model is MORE bullish
            if signal <= 0:
                continue

            break_even = 1 / odds[j]

            # True betting edge versus Bet365 price
            betting_edge = (
                model_probability
                - break_even
            )

            expected_roi = (
                model_probability
                * odds[j]
                - 1
            )

            if j == 0:
                pick = row["home_team"]

            elif j == 1:
                pick = "DRAW"

            else:
                pick = row["away_team"]

            actual_result = row["result"]

            won = (
                actual_result
                == OUTCOME_CODES[j]
            )

            actual_clv = (
                close[i, j]
                - market_probability
            )

            candidate = {

                "date":
                    row["date"],

                "league":
                    row["league"],

                "home_team":
                    row["home_team"],

                "away_team":
                    row["away_team"],

                "pick":
                    pick,

                "pick_code":
                    OUTCOME_CODES[j],

                "odds":
                    odds[j],

                "model_probability":
                    model_probability,

                "market_probability":
                    market_probability,

                "break_even":
                    break_even,

                "signal":
                    signal,

                "betting_edge":
                    betting_edge,

                "expected_roi":
                    expected_roi,

                "actual_clv":
                    actual_clv,

                "actual_result":
                    actual_result,

                "won":
                    won,
            }

            # One candidate per match:
            # choose highest expected ROI
            if (
                best is None
                or
                expected_roi
                > best["expected_roi"]
            ):
                best = candidate

        if best is None:
            continue

        # -----------------------------------------
        # DECISION
        # -----------------------------------------

        if (
            best["signal"] >= MIN_SIGNAL
            and
            best["expected_roi"] >= MIN_EV
        ):

            best["decision"] = "BET"

            if best["won"]:

                best["profit"] = (
                    STAKE
                    * (
                        best["odds"] - 1
                    )
                )

            else:

                best["profit"] = -STAKE

            best["reason"] = "Qualifies"

        else:

            best["decision"] = "SKIP"

            best["profit"] = 0.0

            reasons = []

            if best["signal"] < MIN_SIGNAL:

                reasons.append(
                    "signal < 0.25%"
                )

            if best["expected_roi"] < MIN_EV:

                reasons.append(
                    "EV < 1%"
                )

            best["reason"] = ", ".join(
                reasons
            )

        candidates.append(best)

    return pd.DataFrame(candidates)


def print_summary(bets, title):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if len(bets) == 0:

        print("NO BETS")
        return

    wagered = (
        len(bets) * STAKE
    )

    profit = (
        bets["profit"].sum()
    )

    wins = int(
        bets["won"].sum()
    )

    roi = (
        profit / wagered
    )

    print(
        f"Bets:        {len(bets)}"
    )

    print(
        f"Wins:        {wins}"
    )

    print(
        f"Losses:      {len(bets) - wins}"
    )

    print(
        f"Win rate:    "
        f"{wins / len(bets):.2%}"
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
        f"{bets['expected_roi'].mean():+.2%}"
    )

    print(
        f"Average CLV: "
        f"{bets['actual_clv'].mean():+.2%}"
    )


def historical_simulation(df):

    all_bets = []

    print(
        "\nBET365-ONLY HISTORICAL SIMULATION"
    )

    print(
        "=" * 65
    )

    for (
        season,
        validation_start,
        test_start,
        test_end,
    ) in SEASONS:

        validation_start = (
            pd.Timestamp(validation_start)
        )

        test_start = (
            pd.Timestamp(test_start)
        )

        test_end = (
            pd.Timestamp(test_end)
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

        # -----------------------------------------
        # Train using everything available
        # before test season
        # -----------------------------------------

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

        predicted = predict_probabilities(
            test,
            home_model,
            away_model
        )

        candidates = create_candidates(
            test,
            predicted
        )

        bets = candidates[
            candidates["decision"]
            == "BET"
        ].copy()

        bets["season"] = season

        print_summary(
            bets,
            season
        )

        if len(bets) > 0:

            all_bets.append(bets)

    if not all_bets:

        return pd.DataFrame()

    return pd.concat(
        all_bets,
        ignore_index=True
    )


def last_week_simulation(full_data):

    predictions = pd.read_csv(

        PROCESSED_DIR
        / "last_week_predictions.csv",

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

        "b365_home_odds",
        "b365_draw_odds",
        "b365_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    merged = predictions.merge(

        full_data[
            merge_columns
            + odds_columns
        ],

        on=merge_columns,
        how="left"
    )

    probabilities = merged[
        [
            "model_home",
            "model_draw",
            "model_away",
        ]
    ].to_numpy()

    return create_candidates(
        merged,
        probabilities
    )


def display_last_week(candidates):

    bets = candidates[
        candidates["decision"]
        == "BET"
    ].copy()

    print(
        "\n"
        + "=" * 115
    )

    print(
        "LAST WEEK — BETTING DECISIONS"
    )

    print(
        "=" * 115
    )

    if len(bets) == 0:

        print(
            "\nNO QUALIFYING BETS LAST WEEK."
        )

        print(
            "The model would have stayed out."
        )

    else:

        print(
            "\nBET THESE MATCHES "
            "(retrospective paper simulation):\n"
        )

        show_table(bets)

    # -----------------------------------------
    # Always show closest candidates
    # -----------------------------------------

    print(
        "\n"
        + "-" * 115
    )

    print(
        "CLOSEST SKIPPED CANDIDATES"
    )

    print(
        "-" * 115
    )

    skipped = candidates[
        candidates["decision"]
        == "SKIP"
    ].copy()

    skipped = skipped.sort_values(
        "expected_roi",
        ascending=False
    ).head(10)

    show_table(skipped)


def show_table(df):

    if len(df) == 0:

        print(
            "None"
        )

        return

    display = df.copy()

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

    display["break_even_pct"] = (
        display["break_even"]
        * 100
    )

    display["signal_pct"] = (
        display["signal"]
        * 100
    )

    display["edge_pct"] = (
        display["betting_edge"]
        * 100
    )

    display["ev_pct"] = (
        display["expected_roi"]
        * 100
    )

    display["actual"] = np.where(
        display["won"],
        "WIN",
        "LOSS"
    )

    columns = [

        "date",
        "league",
        "match",
        "pick",
        "decision",
        "odds",
        "model",
        "break_even_pct",
        "signal_pct",
        "edge_pct",
        "ev_pct",
        "reason",
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

                "break_even_pct":
                    lambda x:
                    f"{x:.1f}%",

                "signal_pct":
                    lambda x:
                    f"{x:+.2f}%",

                "edge_pct":
                    lambda x:
                    f"{x:+.2f}%",

                "ev_pct":
                    lambda x:
                    f"{x:+.2f}%",
            }
        )
    )


def run():

    print(
        "\nCONSERVATIVE PAPER-BET RULE"
    )

    print(
        "=" * 65
    )

    print(
        "Price source:           Bet365 only"
    )

    print(
        f"Minimum model signal:  "
        f"{MIN_SIGNAL:.2%}"
    )

    print(
        f"Minimum expected ROI:  "
        f"{MIN_EV:.2%}"
    )

    print(
        f"Stake:                 "
        f"${STAKE:.2f}"
    )

    print(
        "Maximum bets/match:    1"
    )

    raw = pd.read_csv(

        PROCESSED_DIR
        / "matches_advanced_features.csv",

        parse_dates=["date"]
    )

    full_data = raw.copy()

    prepared, _, _ = (
        prepare_data(raw)
    )

    # ==========================================
    # HISTORICAL
    # ==========================================

    historical_bets = (
        historical_simulation(
            prepared
        )
    )

    if len(historical_bets) > 0:

        print_summary(
            historical_bets,
            "ALL BET365-ONLY OOS BETS"
        )

    # ==========================================
    # LAST WEEK
    # ==========================================

    candidates = (
        last_week_simulation(
            full_data
        )
    )

    display_last_week(
        candidates
    )

    recent_bets = candidates[
        candidates["decision"]
        == "BET"
    ]

    print_summary(
        recent_bets,
        "LAST WEEK BET SUMMARY"
    )

    # ==========================================
    # SAVE
    # ==========================================

    historical_bets.to_csv(

        PROCESSED_DIR
        / "bet365_historical_bets.csv",

        index=False
    )

    candidates.to_csv(

        PROCESSED_DIR
        / "last_week_bet_decisions.csv",

        index=False
    )


if __name__ == "__main__":
    run()
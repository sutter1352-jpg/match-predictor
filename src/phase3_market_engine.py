import math

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, LEAGUES

from .match_predictor import (
    load_matches,
    available_teams,
    resolve_team,
    predict_match,
    choose_league,
)

from .polymarket_scanner import (
    find_exact_games,
    get_moneyline_markets,
    parse_array,
    get_order_book,
    best_prices,
    normalize_text,
    canonical_team_name,
)


# ============================================================
# PHASE 3 SETTINGS
# ============================================================

# PAPER MODE ONLY
PAPER_MODE = True

# Minimum difference between our fair probability
# and the executable Polymarket YES ask.
MIN_PAPER_EDGE = 0.02

# Minimum expected return on amount paid.
MIN_PAPER_ROI = 0.04

# Smaller positive edges go on the watchlist.
MIN_WATCH_EDGE = 0.005

# If the standalone football model disagrees with the
# sportsbook anchor by more than this amount, downgrade
# a potential PAPER BET to WATCH.
MAX_MODEL_DISAGREEMENT = 0.05


# ============================================================
# TEAM TEXT MATCHING
# ============================================================

IGNORE_TEAM_WORDS = {
    "fc",
    "cf",
    "afc",
    "club",
    "de",
    "calcio",
    "1901",
}


def team_tokens(
    team
):

    canonical = canonical_team_name(
        team
    )

    return [
        token
        for token in canonical.split()
        if token not in IGNORE_TEAM_WORDS
    ]


def team_in_question(
    team,
    question
):

    question = normalize_text(
        question
    )

    tokens = team_tokens(
        team
    )

    if not tokens:
        return False

    question_tokens = set(
        question.split()
    )

    return all(
        token in question_tokens
        for token in tokens
    )


# ============================================================
# CLASSIFY POLYMARKET MONEYLINE CONTRACT
#
# H = home win
# D = draw
# A = away win
# ============================================================

def classify_market(
    market,
    home_team,
    away_team
):

    question = normalize_text(
        market.get(
            "question",
            ""
        )
    )

    # Draw must be checked first.
    if "draw" in question:
        return "D"

    if (
        "win" in question
        and team_in_question(
            home_team,
            question
        )
    ):
        return "H"

    if (
        "win" in question
        and team_in_question(
            away_team,
            question
        )
    ):
        return "A"

    return None


# ============================================================
# GET YES TOKEN
# ============================================================

def get_yes_token(
    market
):

    outcomes = parse_array(
        market.get(
            "outcomes"
        )
    )

    token_ids = parse_array(
        market.get(
            "clobTokenIds"
        )
    )

    for index, outcome in enumerate(
        outcomes
    ):

        if (
            str(outcome)
            .strip()
            .lower()
            == "yes"
        ):

            if index < len(
                token_ids
            ):

                return token_ids[index]

    return None


# ============================================================
# GET EXECUTABLE YES PRICES
# ============================================================

def get_yes_market_data(
    market
):

    token_id = get_yes_token(
        market
    )

    if token_id is None:

        return {
            "token_id": None,
            "bid": None,
            "ask": None,
            "spread": None,
            "bid_size": None,
            "ask_size": None,
        }

    book = get_order_book(
        token_id
    )

    if book is None:

        return {
            "token_id": token_id,
            "bid": None,
            "ask": None,
            "spread": None,
            "bid_size": None,
            "ask_size": None,
        }

    (
        bid,
        bid_size,
        ask,
        ask_size,
    ) = best_prices(
        book
    )

    spread = None

    if (
        bid is not None
        and ask is not None
    ):

        spread = (
            ask
            - bid
        )

    return {
        "token_id":
            token_id,

        "bid":
            bid,

        "ask":
            ask,

        "spread":
            spread,

        "bid_size":
            bid_size,

        "ask_size":
            ask_size,
    }


# ============================================================
# EXTRACT H / D / A POLYMARKET DATA
# ============================================================

def extract_polymarket_prices(
    event,
    home_team,
    away_team
):

    markets = get_moneyline_markets(
        event
    )

    results = {
        "H": None,
        "D": None,
        "A": None,
    }

    for market in markets:

        outcome = classify_market(
            market,
            home_team,
            away_team
        )

        if outcome is None:
            continue

        market_data = (
            get_yes_market_data(
                market
            )
        )

        entry = {
            "question":
                market.get(
                    "question",
                    ""
                ),

            "market":
                market,

            **market_data,
        }

        # Usually there is only one contract per outcome.
        #
        # If duplicates exist, prefer one that has
        # an executable ask.
        if (
            results[outcome]
            is None
        ):

            results[
                outcome
            ] = entry

        elif (
            results[outcome][
                "ask"
            ]
            is None
            and entry[
                "ask"
            ]
            is not None
        ):

            results[
                outcome
            ] = entry

    return results


# ============================================================
# SPORTSBOOK ODDS
# ============================================================

def ask_for_sportsbook_odds():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CURRENT SPORTSBOOK ODDS"
    )

    print(
        "=" * 72
    )

    print(
        "Enter Home Draw Away."
    )

    print(
        "\nDECIMAL example:"
    )

    print(
        "2.15 3.40 4.20"
    )

    print(
        "\nPERCENTAGE example:"
    )

    print(
        "46 29 27"
    )

    print(
        "or:"
    )

    print(
        "46% 29% 27%"
    )

    print(
        "\nPress Enter to skip."
    )

    value = input(
        "Odds: "
    ).strip()

    if value == "":
        return None

    try:

        parts = value.split()

        if len(parts) != 3:
            raise ValueError

        has_percent_sign = any(
            "%" in part
            for part in parts
        )

        numbers = [
            float(
                part.replace(
                    "%",
                    ""
                )
            )
            for part in parts
        ]

        # ====================================================
        # EXPLICIT % FORMAT
        # ====================================================

        if has_percent_sign:

            if any(
                number <= 0
                or number >= 100
                for number in numbers
            ):
                raise ValueError

            probabilities = np.array(
                numbers,
                dtype=float
            ) / 100.0

            decimal_odds = (
                1.0
                / probabilities
            )

            print(
                "\nPercentage format detected."
            )

            print(
                "Implied probabilities:"
            )

            print(
                f"Home: {probabilities[0]:.1%}"
            )

            print(
                f"Draw: {probabilities[1]:.1%}"
            )

            print(
                f"Away: {probabilities[2]:.1%}"
            )

            print(
                f"Total: "
                f"{probabilities.sum():.1%}"
            )

            return (
                decimal_odds.tolist()
            )

        # ====================================================
        # PERCENTAGES WITHOUT % SIGNS
        #
        # Example:
        #
        # 6 10 86
        #
        # Sum = 102, so these clearly represent percentages.
        # ====================================================

        total = sum(
            numbers
        )

        if (
            all(
                0 < number < 100
                for number in numbers
            )
            and
            80 <= total <= 130
        ):

            probabilities = np.array(
                numbers,
                dtype=float
            ) / 100.0

            decimal_odds = (
                1.0
                / probabilities
            )

            print(
                "\nPercentage format detected."
            )

            print(
                "Implied probabilities:"
            )

            print(
                f"Home: {probabilities[0]:.1%}"
            )

            print(
                f"Draw: {probabilities[1]:.1%}"
            )

            print(
                f"Away: {probabilities[2]:.1%}"
            )

            print(
                f"Total: "
                f"{probabilities.sum():.1%}"
            )

            return (
                decimal_odds.tolist()
            )

        # ====================================================
        # DECIMAL ODDS
        # ====================================================

        if any(
            number <= 1.0
            for number in numbers
        ):
            raise ValueError

        print(
            "\nDecimal format detected."
        )

        return numbers

    except ValueError:

        print(
            "\nInvalid odds."
        )

        print(
            "Use one of these formats:"
        )

        print(
            "Decimal:    2.15 3.40 4.20"
        )

        print(
            "Percentage: 46 29 27"
        )

        print(
            "Percentage: 46% 29% 27%"
        )

        return None


# ============================================================
# REMOVE SPORTSBOOK VIG
# ============================================================

def sportsbook_fair_probabilities(
    odds
):

    implied = np.array(
        [
            1 / odds[0],
            1 / odds[1],
            1 / odds[2],
        ],
        dtype=float
    )

    total = implied.sum()

    fair = (
        implied
        / total
    )

    overround = (
        total
        - 1.0
    )

    return (
        fair,
        overround
    )


# ============================================================
# EXPECTED ROI
#
# Buy YES at price p.
# If true, contract pays $1.
#
# EV per $1 spent:
#
# fair_probability / ask - 1
# ============================================================

def expected_roi(
    fair_probability,
    ask
):

    if (
        ask is None
        or ask <= 0
    ):

        return None

    return (
        fair_probability
        / ask
        - 1.0
    )


# ============================================================
# PAPER DECISION
# ============================================================

def make_decision(
    fair_probability,
    model_probability,
    ask
):

    if ask is None:

        return (
            "NO MARKET",
            "No executable Polymarket ask."
        )

    edge = (
        fair_probability
        - ask
    )

    roi = expected_roi(
        fair_probability,
        ask
    )

    model_disagreement = (
        model_probability
        - fair_probability
    )

    # --------------------------------------------------------
    # Negative or zero edge
    # --------------------------------------------------------

    if edge <= 0:

        return (
            "SKIP",
            "Polymarket ask is at or above fair value."
        )

    # --------------------------------------------------------
    # Strong apparent edge
    # --------------------------------------------------------

    if (
        edge >= MIN_PAPER_EDGE
        and
        roi is not None
        and
        roi >= MIN_PAPER_ROI
    ):

        if (
            model_disagreement
            < -MAX_MODEL_DISAGREEMENT
        ):

            return (
                "WATCH",
                "Sportsbook shows value, but football model "
                "disagrees too strongly."
            )

        return (
            "PAPER BET",
            "Meets paper edge and expected-ROI thresholds."
        )

    # --------------------------------------------------------
    # Small positive edge
    # --------------------------------------------------------

    if edge >= MIN_WATCH_EDGE:

        return (
            "WATCH",
            "Positive edge, but below paper-bet threshold."
        )

    return (
        "SKIP",
        "Edge is too small."
    )


# ============================================================
# DATE INPUT
# ============================================================

def ask_match_date():

    value = input(
        "Match date YYYY-MM-DD: "
    ).strip()

    if value == "":

        return None

    try:

        return pd.Timestamp(
            value
        )

    except ValueError:

        print(
            "Invalid date."
        )

        return None


# ============================================================
# FORMAT
# ============================================================

def fmt_pct(
    value
):

    if value is None:
        return "N/A"

    return (
        f"{value:.1%}"
    )


def fmt_signed_pct(
    value
):

    if value is None:
        return "N/A"

    return (
        f"{value:+.1%}"
    )


def fmt_odds(
    value
):

    if value is None:
        return "N/A"

    return (
        f"{value:.2f}"
    )


# ============================================================
# RUN ONE MATCH
# ============================================================

def analyze_match(
    matches,
    league_code,
    home_team,
    away_team,
    match_date,
    sportsbook_odds
):

    if match_date is None:

        prediction_date = (
            pd.Timestamp.today()
            .normalize()
        )

        requested_date = None

    else:

        prediction_date = (
            match_date.normalize()
        )

        requested_date = (
            prediction_date.date()
        )

    # ========================================================
    # FOOTBALL MODEL
    # ========================================================

    print(
        "\nRunning calibrated football model..."
    )

    prediction = predict_match(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date
    )

    model_probabilities = (
        prediction[
            "probabilities"
        ]
    )

    home_xg = (
        prediction[
            "home_xg"
        ]
    )

    away_xg = (
        prediction[
            "away_xg"
        ]
    )

    # ========================================================
    # POLYMARKET
    # ========================================================

    print(
        "Searching Polymarket for exact fixture..."
    )

    games = find_exact_games(
        home_team,
        away_team,
        requested_date
    )

    if not games:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "NO EXACT POLYMARKET GAME FOUND"
        )

        print(
            "=" * 72
        )

        print(
            f"{home_team} vs {away_team}"
        )

        if requested_date:

            print(
                f"Date: "
                f"{requested_date}"
            )

        return

    event = games[0]

    poly = (
        extract_polymarket_prices(
            event,
            home_team,
            away_team
        )
    )

    # ========================================================
    # SPORTSBOOK FAIR MARKET
    # ========================================================

    sportsbook_fair = None
    overround = None

    if sportsbook_odds is not None:

        (
            sportsbook_fair,
            overround,
        ) = (
            sportsbook_fair_probabilities(
                sportsbook_odds
            )
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n"
        + "=" * 82
    )

    print(
        f"{home_team} vs {away_team}"
    )

    print(
        "=" * 82
    )

    if match_date is not None:

        print(
            f"Match date: "
            f"{match_date.date()}"
        )

    print(
        "\nCALIBRATED FOOTBALL MODEL"
    )

    print(
        "-" * 60
    )

    print(
        f"{home_team:<30}"
        f"{model_probabilities[0]:>10.1%}"
    )

    print(
        f"{'DRAW':<30}"
        f"{model_probabilities[1]:>10.1%}"
    )

    print(
        f"{away_team:<30}"
        f"{model_probabilities[2]:>10.1%}"
    )

    print(
        "\nEXPECTED GOALS"
    )

    print(
        "-" * 60
    )

    print(
        f"{home_team:<30}"
        f"{home_xg:>10.2f}"
    )

    print(
        f"{away_team:<30}"
        f"{away_xg:>10.2f}"
    )

    # ========================================================
    # SPORTSBOOK
    # ========================================================

    if sportsbook_fair is not None:

        print(
            "\nSPORTSBOOK CONSENSUS"
        )

        print(
            "-" * 60
        )

        print(
            f"{'Raw overround':<30}"
            f"{overround:>10.1%}"
        )

        print(
            "\nVig-free fair probabilities:"
        )

        print(
            f"{home_team:<30}"
            f"{sportsbook_fair[0]:>10.1%}"
        )

        print(
            f"{'DRAW':<30}"
            f"{sportsbook_fair[1]:>10.1%}"
        )

        print(
            f"{away_team:<30}"
            f"{sportsbook_fair[2]:>10.1%}"
        )

    else:

        print(
            "\nSPORTSBOOK CONSENSUS"
        )

        print(
            "-" * 60
        )

        print(
            "No sportsbook odds entered."
        )

    # ========================================================
    # POLYMARKET
    # ========================================================

    print(
        "\nPOLYMARKET EXECUTABLE YES MARKET"
    )

    print(
        "-" * 82
    )

    print(
        f"{'OUTCOME':<25}"
        f"{'BID':>10}"
        f"{'ASK':>10}"
        f"{'SPREAD':>10}"
    )

    print(
        "-" * 82
    )

    outcome_names = [
        home_team,
        "DRAW",
        away_team,
    ]

    outcome_codes = [
        "H",
        "D",
        "A",
    ]

    for name, code in zip(
        outcome_names,
        outcome_codes
    ):

        data = poly[
            code
        ]

        if data is None:

            bid = None
            ask = None
            spread = None

        else:

            bid = data[
                "bid"
            ]

            ask = data[
                "ask"
            ]

            spread = data[
                "spread"
            ]

        print(
            f"{name:<25}"
            f"{fmt_pct(bid):>10}"
            f"{fmt_pct(ask):>10}"
            f"{fmt_pct(spread):>10}"
        )

    # ========================================================
    # EDGE ENGINE
    # ========================================================

    print(
        "\n"
        + "=" * 82
    )

    print(
        "PHASE 3 PAPER DECISION ENGINE"
    )

    print(
        "=" * 82
    )

    if sportsbook_fair is None:

        print(
            "\nSportsbook odds are required "
            "for fair-value decisions."
        )

        print(
            "Polymarket prices and football "
            "prediction were loaded successfully."
        )

        return

    rows = []

    for index, (
        name,
        code,
    ) in enumerate(
        zip(
            outcome_names,
            outcome_codes
        )
    ):

        model_probability = (
            float(
                model_probabilities[
                    index
                ]
            )
        )

        fair_probability = (
            float(
                sportsbook_fair[
                    index
                ]
            )
        )

        market_data = poly[
            code
        ]

        if market_data is None:

            ask = None
            bid = None
            spread = None

        else:

            ask = (
                market_data[
                    "ask"
                ]
            )

            bid = (
                market_data[
                    "bid"
                ]
            )

            spread = (
                market_data[
                    "spread"
                ]
            )

        if ask is not None:

            edge = (
                fair_probability
                - ask
            )

            roi = expected_roi(
                fair_probability,
                ask
            )

        else:

            edge = None
            roi = None

        model_vs_market = (
            model_probability
            - fair_probability
        )

        (
            decision,
            reason,
        ) = make_decision(
            fair_probability,
            model_probability,
            ask
        )

        rows.append({
            "Outcome":
                name,

            "Football Model":
                model_probability,

            "Sportsbook Fair":
                fair_probability,

            "Model vs Book":
                model_vs_market,

            "Polymarket Bid":
                bid,

            "Polymarket Ask":
                ask,

            "Spread":
                spread,

            "Edge":
                edge,

            "Expected ROI":
                roi,

            "Decision":
                decision,

            "Reason":
                reason,
        })

    # ========================================================
    # SIMPLE CONSOLE TABLE
    # ========================================================

    print(
        "\n"
        f"{'OUTCOME':<22}"
        f"{'MODEL':>9}"
        f"{'FAIR':>9}"
        f"{'ASK':>9}"
        f"{'EDGE':>9}"
        f"{'ROI':>9}"
        f"{'DECISION':>14}"
    )

    print(
        "-" * 82
    )

    for row in rows:

        print(
            f"{row['Outcome']:<22}"
            f"{fmt_pct(row['Football Model']):>9}"
            f"{fmt_pct(row['Sportsbook Fair']):>9}"
            f"{fmt_pct(row['Polymarket Ask']):>9}"
            f"{fmt_signed_pct(row['Edge']):>9}"
            f"{fmt_signed_pct(row['Expected ROI']):>9}"
            f"{row['Decision']:>14}"
        )

    # ========================================================
    # DETAILS
    # ========================================================

    print(
        "\nDECISION DETAILS"
    )

    print(
        "-" * 82
    )

    for row in rows:

        print(
            f"\n{row['Outcome']}"
        )

        print(
            f"  Football model:      "
            f"{fmt_pct(row['Football Model'])}"
        )

        print(
            f"  Sportsbook fair:     "
            f"{fmt_pct(row['Sportsbook Fair'])}"
        )

        print(
            f"  Model vs sportsbook: "
            f"{fmt_signed_pct(row['Model vs Book'])}"
        )

        print(
            f"  Polymarket bid:      "
            f"{fmt_pct(row['Polymarket Bid'])}"
        )

        print(
            f"  Polymarket ask:      "
            f"{fmt_pct(row['Polymarket Ask'])}"
        )

        print(
            f"  Edge:                "
            f"{fmt_signed_pct(row['Edge'])}"
        )

        print(
            f"  Expected ROI:        "
            f"{fmt_signed_pct(row['Expected ROI'])}"
        )

        print(
            f"  Decision:            "
            f"{row['Decision']}"
        )

        print(
            f"  Reason:              "
            f"{row['Reason']}"
        )

    # ========================================================
    # BEST OPPORTUNITY
    # ========================================================

    valid_rows = [
        row
        for row in rows
        if row[
            "Edge"
        ] is not None
    ]

    if valid_rows:

        best = max(
            valid_rows,
            key=lambda row:
            row["Edge"]
        )

        print(
            "\n"
            + "=" * 82
        )

        print(
            "BEST CURRENT OPPORTUNITY"
        )

        print(
            "=" * 82
        )

        print(
            f"Outcome:       "
            f"{best['Outcome']}"
        )

        print(
            f"Fair value:    "
            f"{fmt_pct(best['Sportsbook Fair'])}"
        )

        print(
            f"Polymarket ask:"
            f" {fmt_pct(best['Polymarket Ask'])}"
        )

        print(
            f"Edge:          "
            f"{fmt_signed_pct(best['Edge'])}"
        )

        print(
            f"Expected ROI:  "
            f"{fmt_signed_pct(best['Expected ROI'])}"
        )

        print(
            f"Decision:      "
            f"{best['Decision']}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = pd.DataFrame(
        rows
    )

    output["date"] = (
        prediction_date.date()
    )

    output["league_code"] = (
        league_code
    )

    output["home_team"] = (
        home_team
    )

    output["away_team"] = (
        away_team
    )

    output_path = (
        PROCESSED_DIR
        /
        "phase3_latest_scan.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        "\n"
        + "=" * 82
    )

    print(
        "PAPER MODE ONLY"
    )

    print(
        "No Polymarket orders were placed."
    )

    print(
        "=" * 82
    )


# ============================================================
# MAIN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 82
    )

    print(
        "PHASE 3B — SOCCER MARKET ENGINE"
    )

    print(
        "=" * 82
    )

    print(
        "\nPAPER MODE ONLY"
    )

    print(
        "This program will NOT place bets."
    )

    matches = load_matches()

    print(
        f"\nMatches loaded: "
        f"{len(matches):,}"
    )

    while True:

        # ====================================================
        # LEAGUE
        # ====================================================

        league_code = (
            choose_league()
        )

        if league_code is None:

            print(
                "Invalid league."
            )

            continue

        teams = available_teams(
            matches,
            league_code
        )

        print(
            f"\n{LEAGUES[league_code]}"
        )

        print(
            f"Current teams found: "
            f"{len(teams)}"
        )

        # ====================================================
        # HOME TEAM
        # ====================================================

        home_input = input(
            "\nHome team: "
        ).strip()

        home_team = resolve_team(
            home_input,
            teams
        )

        if home_team is None:

            print(
                "Home team not found."
            )

            continue

        # ====================================================
        # AWAY TEAM
        # ====================================================

        away_input = input(
            "Away team: "
        ).strip()

        away_team = resolve_team(
            away_input,
            teams
        )

        if away_team is None:

            print(
                "Away team not found."
            )

            continue

        if (
            home_team
            == away_team
        ):

            print(
                "Teams cannot be the same."
            )

            continue

        # ====================================================
        # DATE
        # ====================================================

        match_date = ask_match_date()

        # ====================================================
        # SPORTSBOOK
        # ====================================================

        sportsbook_odds = (
            ask_for_sportsbook_odds()
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        analyze_match(
            matches,
            league_code,
            home_team,
            away_team,
            match_date,
            sportsbook_odds
        )

        again = input(
            "\nAnalyze another match? "
            "(y/n): "
        ).strip().lower()

        if again not in {
            "y",
            "yes"
        }:

            break

    print(
        "\nPhase 3 engine closed."
    )


if __name__ == "__main__":
    run()
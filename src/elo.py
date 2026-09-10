from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR


INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0


def expected_home_score(home_elo: float, away_elo: float) -> float:
    """
    Expected home score on the Elo scale.
    A win counts as 1, draw as 0.5, loss as 0.
    """
    adjusted_home = home_elo + HOME_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def actual_home_score(result: str) -> float:
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    return 0.0


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT:
    Elo values written onto a match are the ratings BEFORE that match.
    This prevents target leakage.
    """
    df = df.sort_values(["date", "match_id"]).copy()

    # Separate team ratings by league for the domestic Phase-1 baseline.
    ratings = defaultdict(lambda: INITIAL_ELO)

    home_pre = []
    away_pre = []
    diff_pre = []
    home_expected = []

    for row in df.itertuples(index=False):
        home_key = (row.league_code, row.home_team)
        away_key = (row.league_code, row.away_team)

        h = ratings[home_key]
        a = ratings[away_key]

        exp_h = expected_home_score(h, a)

        home_pre.append(h)
        away_pre.append(a)
        diff_pre.append((h + HOME_ADVANTAGE) - a)
        home_expected.append(exp_h)

        actual_h = actual_home_score(row.result)
        change = K_FACTOR * (actual_h - exp_h)

        ratings[home_key] = h + change
        ratings[away_key] = a - change

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    df["elo_diff_home_adjusted"] = diff_pre
    df["elo_expected_home_score"] = home_expected

    return df


def run() -> pd.DataFrame:
    input_path = PROCESSED_DIR / "matches_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "matches_clean.csv not found. Run: python -m src.build_dataset"
        )

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = add_elo_features(df)

    output_path = PROCESSED_DIR / "matches_with_elo.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved Elo dataset -> {output_path}")
    print(
        df[
            [
                "date", "league", "home_team", "away_team", "result",
                "home_elo_pre", "away_elo_pre",
                "elo_diff_home_adjusted", "elo_expected_home_score",
            ]
        ].tail(10).to_string(index=False)
    )

    return df


if __name__ == "__main__":
    run()

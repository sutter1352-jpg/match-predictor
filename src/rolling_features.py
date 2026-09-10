import pandas as pd
from collections import defaultdict, deque

from .config import PROCESSED_DIR


WINDOW = 5


def add_rolling_features(df):
    df = df.sort_values(["date", "match_id"]).copy()

    team_history = defaultdict(lambda: deque(maxlen=WINDOW))

    features = {
        "home_points_last5": [],
        "away_points_last5": [],
        "home_goals_for_last5": [],
        "away_goals_for_last5": [],
        "home_goals_against_last5": [],
        "away_goals_against_last5": [],
    }

    for row in df.itertuples(index=False):
        home_key = (row.league_code, row.home_team)
        away_key = (row.league_code, row.away_team)

        home_history = list(team_history[home_key])
        away_history = list(team_history[away_key])

        def summarize(history):
            if not history:
                return 0, 0, 0

            points = sum(x["points"] for x in history)
            goals_for = sum(x["goals_for"] for x in history)
            goals_against = sum(x["goals_against"] for x in history)

            return points, goals_for, goals_against

        hp, hgf, hga = summarize(home_history)
        ap, agf, aga = summarize(away_history)

        features["home_points_last5"].append(hp)
        features["away_points_last5"].append(ap)

        features["home_goals_for_last5"].append(hgf)
        features["away_goals_for_last5"].append(agf)

        features["home_goals_against_last5"].append(hga)
        features["away_goals_against_last5"].append(aga)

        if row.result == "H":
            home_points = 3
            away_points = 0
        elif row.result == "A":
            home_points = 0
            away_points = 3
        else:
            home_points = 1
            away_points = 1

        team_history[home_key].append({
            "points": home_points,
            "goals_for": row.home_goals,
            "goals_against": row.away_goals,
        })

        team_history[away_key].append({
            "points": away_points,
            "goals_for": row.away_goals,
            "goals_against": row.home_goals,
        })

    for column, values in features.items():
        df[column] = values

    return df


def run():
    path = PROCESSED_DIR / "matches_with_elo.csv"

    df = pd.read_csv(path, parse_dates=["date"])

    df = add_rolling_features(df)

    output_path = PROCESSED_DIR / "matches_with_features.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved -> {output_path}")

    print(
        df[
            [
                "date",
                "home_team",
                "away_team",
                "home_points_last5",
                "away_points_last5",
                "home_goals_for_last5",
                "away_goals_for_last5",
            ]
        ].tail(10)
    )


if __name__ == "__main__":
    run()
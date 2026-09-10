import pandas as pd

from .config import PROCESSED_DIR


def validate() -> None:
    path = PROCESSED_DIR / "matches_with_elo.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    errors = []

    if df["match_id"].duplicated().any():
        errors.append("Duplicate match_id values found.")

    if not set(df["result"].dropna().unique()).issubset({"H", "D", "A"}):
        errors.append("Unexpected result labels found.")

    if df[["home_team", "away_team", "date"]].isna().any().any():
        errors.append("Missing team/date values found.")

    # Leakage sanity check: the very first observed match for each league
    # should begin from the initial Elo level.
    first_rows = (
        df.sort_values(["date", "match_id"])
          .groupby("league_code", as_index=False)
          .first()
    )

    if not ((first_rows["home_elo_pre"] == 1500.0) &
            (first_rows["away_elo_pre"] == 1500.0)).all():
        errors.append("Unexpected initial Elo values.")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Leagues: {df['league'].nunique()}")
    print("\nResult distribution:")
    print(df["result"].value_counts(normalize=True).rename("share").round(4))


if __name__ == "__main__":
    validate()

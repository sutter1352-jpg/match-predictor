import hashlib
from pathlib import Path

import pandas as pd

from .config import LEAGUES, PROCESSED_DIR, RAW_DIR


# ============================================================
# COLUMNS WE WANT TO KEEP FROM FOOTBALL-DATA
# ============================================================

KEEP_COLUMNS = [

    # Match identity
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",

    # Full-time result
    "FTHG",
    "FTAG",
    "FTR",

    # Half-time result
    "HTHG",
    "HTAG",
    "HTR",

    # Match statistics
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",

    # ========================================================
    # EARLIER / STANDARD BOOKMAKER ODDS
    # ========================================================

    "B365H",
    "B365D",
    "B365A",

    "PSH",
    "PSD",
    "PSA",

    "MaxH",
    "MaxD",
    "MaxA",

    "AvgH",
    "AvgD",
    "AvgA",

    # ========================================================
    # CLOSING BOOKMAKER ODDS
    # ========================================================

    "B365CH",
    "B365CD",
    "B365CA",

    "PSCH",
    "PSCD",
    "PSCA",

    "MaxCH",
    "MaxCD",
    "MaxCA",

    "AvgCH",
    "AvgCD",
    "AvgCA",
]


# ============================================================
# COLUMN NAMES WE WILL USE
# ============================================================

RENAME_COLUMNS = {

    # Match identity
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",

    # Results
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",

    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",

    # Match statistics
    "HS": "home_shots",
    "AS": "away_shots",

    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",

    "HF": "home_fouls",
    "AF": "away_fouls",

    "HC": "home_corners",
    "AC": "away_corners",

    "HY": "home_yellow",
    "AY": "away_yellow",

    "HR": "home_red",
    "AR": "away_red",

    # ========================================================
    # EARLIER / STANDARD ODDS
    # ========================================================

    "B365H": "b365_home_odds",
    "B365D": "b365_draw_odds",
    "B365A": "b365_away_odds",

    "PSH": "pinnacle_home_odds",
    "PSD": "pinnacle_draw_odds",
    "PSA": "pinnacle_away_odds",

    "MaxH": "max_home_odds",
    "MaxD": "max_draw_odds",
    "MaxA": "max_away_odds",

    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",

    # ========================================================
    # CLOSING ODDS
    # ========================================================

    "B365CH": "b365_close_home_odds",
    "B365CD": "b365_close_draw_odds",
    "B365CA": "b365_close_away_odds",

    "PSCH": "pinnacle_close_home_odds",
    "PSCD": "pinnacle_close_draw_odds",
    "PSCA": "pinnacle_close_away_odds",

    "MaxCH": "max_close_home_odds",
    "MaxCD": "max_close_draw_odds",
    "MaxCA": "max_close_away_odds",

    "AvgCH": "avg_close_home_odds",
    "AvgCD": "avg_close_draw_odds",
    "AvgCA": "avg_close_away_odds",
}


# ============================================================
# DATE PARSING
# ============================================================

def parse_date_value(value):
    """
    Football-Data files contain different date formats
    depending on the season.

    Examples:
        08/08/15
        08/08/2015
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%d/%m/%Y",
        "%d/%m/%y",
    ]

    for date_format in formats:

        try:
            return pd.to_datetime(
                value,
                format=date_format
            )

        except (ValueError, TypeError):
            pass

    return pd.NaT


# ============================================================
# MATCH ID
# ============================================================

def make_match_id(row):

    value = (
        f"{row['date'].date()}|"
        f"{row['league_code']}|"
        f"{row['home_team']}|"
        f"{row['away_team']}"
    )

    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# LOAD ONE CSV
# ============================================================

def load_one(path: Path):

    # File name example:
    #
    # 2526_E0.csv

    season, league_code = (
        path.stem.split("_", 1)
    )

    try:

        df = pd.read_csv(
            path,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            path,
            encoding="latin-1"
        )

    # --------------------------------------------------------
    # Make sure KEEP_COLUMNS itself cannot create duplicates
    # --------------------------------------------------------

    available_columns = list(
        dict.fromkeys(
            column
            for column in KEEP_COLUMNS
            if column in df.columns
        )
    )

    df = df[
        available_columns
    ].copy()

    # --------------------------------------------------------
    # Add competition information
    # --------------------------------------------------------

    df["season"] = str(season)

    df["league_code"] = (
        league_code
    )

    df["league"] = LEAGUES.get(
        league_code,
        league_code
    )

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    df = df.rename(
        columns=RENAME_COLUMNS
    )

    # Extra protection against duplicate columns.
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()

    # --------------------------------------------------------
    # Make sure essential columns exist
    # --------------------------------------------------------

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:

        print(
            f"SKIP {path.name}: "
            f"missing required columns "
            f"{missing_required}"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    df["date"] = (
        df["date"]
        .apply(parse_date_value)
    )

    # --------------------------------------------------------
    # Convert goals to numeric
    # --------------------------------------------------------

    df["home_goals"] = pd.to_numeric(
        df["home_goals"],
        errors="coerce"
    )

    df["away_goals"] = pd.to_numeric(
        df["away_goals"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean team names
    # --------------------------------------------------------

    df["home_team"] = (
        df["home_team"]
        .astype("string")
        .str.strip()
    )

    df["away_team"] = (
        df["away_team"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Keep completed matches only
    # --------------------------------------------------------

    df = df[
        df["date"].notna()
        &
        df["home_team"].notna()
        &
        df["away_team"].notna()
        &
        df["home_goals"].notna()
        &
        df["away_goals"].notna()
        &
        df["result"].isin(
            ["H", "D", "A"]
        )
    ].copy()

    # --------------------------------------------------------
    # Create unique match ID
    # --------------------------------------------------------

    df["match_id"] = (
        df.apply(
            make_match_id,
            axis=1
        )
    )

    return df


# ============================================================
# BUILD FULL DATASET
# ============================================================

def build_dataset():

    paths = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not paths:

        raise FileNotFoundError(
            "No CSV files found in data/raw.\n"
            "Run:\n"
            "python -m src.download_data"
        )

    frames = []

    for path in paths:

        print(
            f"Reading {path.name}"
        )

        frame = load_one(
            path
        )

        if not frame.empty:

            frames.append(
                frame
            )

    if not frames:

        raise RuntimeError(
            "No valid match files were loaded."
        )

    # --------------------------------------------------------
    # Combine all leagues / seasons
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # Remove duplicate matches
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["match_id"]
        )
        .sort_values(
            [
                "date",
                "league_code",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # TARGET LABELS
    # ========================================================

    df[
        "target_home_win"
    ] = (
        df["result"] == "H"
    ).astype(int)

    df[
        "target_draw"
    ] = (
        df["result"] == "D"
    ).astype(int)

    df[
        "target_away_win"
    ] = (
        df["result"] == "A"
    ).astype(int)

    # ========================================================
    # SAVE
    # ========================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        PROCESSED_DIR
        / "matches_clean.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        "\n=================================="
    )

    print(
        "DATASET BUILD COMPLETE"
    )

    print(
        "=================================="
    )

    print(
        f"Total matches: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"-> "
        f"{df['date'].max().date()}"
    )

    print(
        "\nMatches by league:"
    )

    print(
        df.groupby("league")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nSaved ->"
    )

    print(
        output
    )

    # ========================================================
    # CHECK ODDS COLUMNS
    # ========================================================

    print(
        "\nOdds columns available:"
    )

    odds_columns = [
        column
        for column in df.columns
        if "odds" in column.lower()
    ]

    for column in odds_columns:

        available_count = (
            df[column]
            .notna()
            .sum()
        )

        print(
            f"{column}: "
            f"{available_count:,}"
        )

    return df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_dataset()
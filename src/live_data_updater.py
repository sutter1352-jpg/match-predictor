from io import BytesIO
import hashlib

import pandas as pd
import requests

from .config import LEAGUES, PROCESSED_DIR


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/{season}/{league}.csv"
)

REQUEST_TIMEOUT = 30

MATCHES_FILE = (
    PROCESSED_DIR
    / "matches_clean.csv"
)

BACKUP_FILE = (
    PROCESSED_DIR
    / "matches_clean_backup.csv"
)


# ============================================================
# FOOTBALL-DATA -> OUR DATABASE COLUMNS
# ============================================================

COLUMN_MAP = {

    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    "Date":
        "date",

    "HomeTeam":
        "home_team",

    "AwayTeam":
        "away_team",

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    "FTHG":
        "home_goals",

    "FTAG":
        "away_goals",

    "FTR":
        "result",

    # --------------------------------------------------------
    # SHOTS
    # --------------------------------------------------------

    "HS":
        "home_shots",

    "AS":
        "away_shots",

    # --------------------------------------------------------
    # SHOTS ON TARGET
    # --------------------------------------------------------

    "HST":
        "home_shots_on_target",

    "AST":
        "away_shots_on_target",

    # --------------------------------------------------------
    # CORNERS
    # --------------------------------------------------------

    "HC":
        "home_corners",

    "AC":
        "away_corners",

    # --------------------------------------------------------
    # FOULS
    # --------------------------------------------------------

    "HF":
        "home_fouls",

    "AF":
        "away_fouls",

    # --------------------------------------------------------
    # YELLOW CARDS
    # --------------------------------------------------------

    "HY":
        "home_yellow_cards",

    "AY":
        "away_yellow_cards",

    # --------------------------------------------------------
    # RED CARDS
    # --------------------------------------------------------

    "HR":
        "home_red_cards",

    "AR":
        "away_red_cards",

    # --------------------------------------------------------
    # BET365
    # --------------------------------------------------------

    "B365H":
        "b365_home_odds",

    "B365D":
        "b365_draw_odds",

    "B365A":
        "b365_away_odds",

    # --------------------------------------------------------
    # PINNACLE
    # --------------------------------------------------------

    "PSH":
        "pinnacle_home_odds",

    "PSD":
        "pinnacle_draw_odds",

    "PSA":
        "pinnacle_away_odds",

    # --------------------------------------------------------
    # MAXIMUM MARKET ODDS
    # --------------------------------------------------------

    "MaxH":
        "max_home_odds",

    "MaxD":
        "max_draw_odds",

    "MaxA":
        "max_away_odds",

    # --------------------------------------------------------
    # AVERAGE MARKET ODDS
    # --------------------------------------------------------

    "AvgH":
        "avg_home_odds",

    "AvgD":
        "avg_draw_odds",

    "AvgA":
        "avg_away_odds",

    # --------------------------------------------------------
    # BET365 CLOSING
    # --------------------------------------------------------

    "B365CH":
        "b365_close_home_odds",

    "B365CD":
        "b365_close_draw_odds",

    "B365CA":
        "b365_close_away_odds",

    # --------------------------------------------------------
    # PINNACLE CLOSING
    # --------------------------------------------------------

    "PSCH":
        "pinnacle_close_home_odds",

    "PSCD":
        "pinnacle_close_draw_odds",

    "PSCA":
        "pinnacle_close_away_odds",

    # --------------------------------------------------------
    # MAXIMUM CLOSING
    # --------------------------------------------------------

    "MaxCH":
        "max_close_home_odds",

    "MaxCD":
        "max_close_draw_odds",

    "MaxCA":
        "max_close_away_odds",

    # --------------------------------------------------------
    # AVERAGE CLOSING
    # --------------------------------------------------------

    "AvgCH":
        "avg_close_home_odds",

    "AvgCD":
        "avg_close_draw_odds",

    "AvgCA":
        "avg_close_away_odds",
}


# ============================================================
# CURRENT EUROPEAN SEASON
#
# September 2026 -> 2627
# February 2027  -> 2627
# ============================================================

def get_current_season_code(
    today=None
):

    if today is None:

        today = (
            pd.Timestamp.today()
        )

    today = pd.Timestamp(
        today
    )

    if today.month >= 7:

        start_year = (
            today.year
        )

    else:

        start_year = (
            today.year
            - 1
        )

    end_year = (
        start_year
        + 1
    )

    return (
        f"{str(start_year)[-2:]}"
        f"{str(end_year)[-2:]}"
    )


# ============================================================
# DATE PARSER
# ============================================================

def parse_dates(
    series
):

    result = pd.to_datetime(
        series,
        format="%d/%m/%Y",
        errors="coerce"
    )

    missing = (
        result.isna()
    )

    if missing.any():

        result.loc[
            missing
        ] = pd.to_datetime(
            series.loc[
                missing
            ],
            format="%d/%m/%y",
            errors="coerce"
        )

    return result


# ============================================================
# DOWNLOAD ONE LEAGUE
# ============================================================

def download_league(
    league_code,
    season
):

    url = BASE_URL.format(
        season=season,
        league=league_code
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    raw = pd.read_csv(
        BytesIO(
            response.content
        )
    )

    required_columns = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    }

    missing = (
        required_columns
        - set(
            raw.columns
        )
    )

    if missing:

        raise ValueError(
            "Missing expected columns: "
            f"{sorted(missing)}"
        )

    return raw


# ============================================================
# CREATE MATCH ID
# ============================================================

def make_match_id(
    league_code,
    date,
    home_team,
    away_team
):

    date = pd.Timestamp(
        date
    )

    key = (
        f"{league_code}|"
        f"{date.strftime('%Y-%m-%d')}|"
        f"{home_team}|"
        f"{away_team}"
    )

    return hashlib.sha1(
        key.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# BUILD CURRENT SEASON ROWS
# ============================================================

def build_current_rows(
    raw,
    league_code,
    season,
    template_columns
):

    raw = (
        raw.copy()
    )

    raw[
        "_parsed_date"
    ] = parse_dates(
        raw[
            "Date"
        ]
    )

    home_goals = pd.to_numeric(
        raw[
            "FTHG"
        ],
        errors="coerce"
    )

    away_goals = pd.to_numeric(
        raw[
            "FTAG"
        ],
        errors="coerce"
    )

    # ========================================================
    # ONLY COMPLETED MATCHES
    # ========================================================

    completed = raw[
        raw[
            "_parsed_date"
        ].notna()
        &
        home_goals.notna()
        &
        away_goals.notna()
    ].copy()

    if len(completed) == 0:

        return pd.DataFrame(
            columns=template_columns
        )

    # ========================================================
    # IMPORTANT:
    #
    # Force object dtype here.
    #
    # This prevents errors such as:
    #
    # Invalid value ['2627' ...] for dtype int64
    #
    # We normalize the relevant fields afterward.
    # ========================================================

    result = pd.DataFrame(
        {
            column:
                pd.Series(
                    [None]
                    * len(completed),
                    dtype="object"
                )
            for column
            in template_columns
        }
    )

    completed = (
        completed
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # PRESERVE RAW COLUMN IF OUR DATABASE HAS SAME NAME
    # ========================================================

    for source_column in (
        completed.columns
    ):

        if source_column in (
            result.columns
        ):

            result[
                source_column
            ] = (
                completed[
                    source_column
                ]
                .astype(
                    "object"
                )
                .values
            )

    # ========================================================
    # COPY STANDARDIZED COLUMNS
    # ========================================================

    for (
        source_column,
        target_column,
    ) in COLUMN_MAP.items():

        if (
            source_column
            not in completed.columns
        ):

            continue

        if (
            target_column
            not in result.columns
        ):

            continue

        if (
            source_column
            == "Date"
        ):

            result[
                target_column
            ] = (
                completed[
                    "_parsed_date"
                ]
                .astype(
                    "object"
                )
                .values
            )

        else:

            result[
                target_column
            ] = (
                completed[
                    source_column
                ]
                .astype(
                    "object"
                )
                .values
            )

    # ========================================================
    # CORE MATCH INFORMATION
    # ========================================================

    if (
        "date"
        in result.columns
    ):

        result[
            "date"
        ] = (
            completed[
                "_parsed_date"
            ]
            .astype(
                "object"
            )
            .values
        )

    if (
        "home_team"
        in result.columns
    ):

        result[
            "home_team"
        ] = (
            completed[
                "HomeTeam"
            ]
            .astype(
                "object"
            )
            .values
        )

    if (
        "away_team"
        in result.columns
    ):

        result[
            "away_team"
        ] = (
            completed[
                "AwayTeam"
            ]
            .astype(
                "object"
            )
            .values
        )

    # ========================================================
    # LEAGUE
    # ========================================================

    if (
        "league_code"
        in result.columns
    ):

        result[
            "league_code"
        ] = league_code

    if (
        "league"
        in result.columns
    ):

        result[
            "league"
        ] = LEAGUES[
            league_code
        ]

    if (
        "league_name"
        in result.columns
    ):

        result[
            "league_name"
        ] = LEAGUES[
            league_code
        ]

    # ========================================================
    # SEASON
    #
    # IMPORTANT FIX:
    # store it as an integer, not "2627" text.
    # ========================================================

    if (
        "season"
        in result.columns
    ):

        result[
            "season"
        ] = int(
            season
        )

    # ========================================================
    # MATCH IDS
    # ========================================================

    if (
        "match_id"
        in result.columns
    ):

        match_ids = []

        for _, row in (
            completed.iterrows()
        ):

            match_ids.append(
                make_match_id(
                    league_code,
                    row[
                        "_parsed_date"
                    ],
                    row[
                        "HomeTeam"
                    ],
                    row[
                        "AwayTeam"
                    ]
                )
            )

        result[
            "match_id"
        ] = match_ids

    # ========================================================
    # ENSURE GOALS ARE NUMERIC
    # ========================================================

    for column in [
        "home_goals",
        "away_goals",
    ]:

        if column in (
            result.columns
        ):

            result[
                column
            ] = pd.to_numeric(
                result[
                    column
                ],
                errors="coerce"
            )

    result[
        "date"
    ] = pd.to_datetime(
        result[
            "date"
        ],
        errors="coerce"
    )

    return (
        result
        .reset_index(
            drop=True
        )
    )


# ============================================================
# UNIQUE MATCH KEY
# ============================================================

def create_match_keys(
    dataframe
):

    dates = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="coerce"
    ).dt.strftime(
        "%Y-%m-%d"
    )

    return (
        dataframe[
            "league_code"
        ]
        .astype(str)
        .str.strip()
        +
        "|"
        +
        dates.astype(str)
        +
        "|"
        +
        dataframe[
            "home_team"
        ]
        .astype(str)
        .str.strip()
        +
        "|"
        +
        dataframe[
            "away_team"
        ]
        .astype(str)
        .str.strip()
    )


# ============================================================
# MERGE DATABASES
#
# This version deliberately uses object dtype while merging.
#
# That prevents pandas dtype conflicts between:
#
# int64
# float64
# strings
# nullable values
# ============================================================

def merge_matches(
    existing,
    incoming
):

    existing = (
        existing.copy()
    )

    incoming = (
        incoming.copy()
    )

    # Ensure incoming has exactly same columns.
    incoming = incoming.reindex(
        columns=existing.columns
    )

    existing[
        "date"
    ] = pd.to_datetime(
        existing[
            "date"
        ],
        errors="coerce"
    )

    incoming[
        "date"
    ] = pd.to_datetime(
        incoming[
            "date"
        ],
        errors="coerce"
    )

    # ========================================================
    # MAKE MERGE TYPE-SAFE
    # ========================================================

    existing = (
        existing.astype(
            "object"
        )
    )

    incoming = (
        incoming.astype(
            "object"
        )
    )

    # ========================================================
    # KEYS
    # ========================================================

    existing[
        "_update_key"
    ] = create_match_keys(
        existing
    )

    incoming[
        "_update_key"
    ] = create_match_keys(
        incoming
    )

    existing = (
        existing
        .drop_duplicates(
            subset=[
                "_update_key"
            ],
            keep="last"
        )
        .set_index(
            "_update_key"
        )
    )

    incoming = (
        incoming
        .drop_duplicates(
            subset=[
                "_update_key"
            ],
            keep="last"
        )
        .set_index(
            "_update_key"
        )
    )

    existing_keys = set(
        existing.index
    )

    incoming_keys = set(
        incoming.index
    )

    new_keys = (
        incoming_keys
        - existing_keys
    )

    common_keys = (
        incoming_keys
        &
        existing_keys
    )

    # ========================================================
    # REFRESH EXISTING MATCHES
    #
    # Only overwrite when the incoming value is NOT missing.
    # ========================================================

    if common_keys:

        for key in (
            common_keys
        ):

            incoming_row = (
                incoming.loc[
                    key
                ]
            )

            for column in (
                incoming.columns
            ):

                value = (
                    incoming_row[
                        column
                    ]
                )

                if pd.notna(
                    value
                ):

                    existing.at[
                        key,
                        column
                    ] = value

    # ========================================================
    # ADD NEW MATCHES
    # ========================================================

    if new_keys:

        new_rows = (
            incoming.loc[
                list(
                    new_keys
                )
            ]
        )

        combined = pd.concat(
            [
                existing,
                new_rows,
            ],
            axis=0
        )

    else:

        combined = (
            existing
        )

    combined = (
        combined
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # NORMALIZE IMPORTANT DTYPES
    # ========================================================

    combined[
        "date"
    ] = pd.to_datetime(
        combined[
            "date"
        ],
        errors="coerce"
    )

    if (
        "season"
        in combined.columns
    ):

        combined[
            "season"
        ] = pd.to_numeric(
            combined[
                "season"
            ],
            errors="coerce"
        ).astype(
            "Int64"
        )

    numeric_columns = [

        "home_goals",
        "away_goals",

        "home_shots",
        "away_shots",

        "home_shots_on_target",
        "away_shots_on_target",

        "home_corners",
        "away_corners",

        "home_fouls",
        "away_fouls",

        "home_yellow_cards",
        "away_yellow_cards",

        "home_red_cards",
        "away_red_cards",

        "b365_home_odds",
        "b365_draw_odds",
        "b365_away_odds",

        "pinnacle_home_odds",
        "pinnacle_draw_odds",
        "pinnacle_away_odds",

        "max_home_odds",
        "max_draw_odds",
        "max_away_odds",

        "avg_home_odds",
        "avg_draw_odds",
        "avg_away_odds",

        "b365_close_home_odds",
        "b365_close_draw_odds",
        "b365_close_away_odds",

        "pinnacle_close_home_odds",
        "pinnacle_close_draw_odds",
        "pinnacle_close_away_odds",

        "max_close_home_odds",
        "max_close_draw_odds",
        "max_close_away_odds",

        "avg_close_home_odds",
        "avg_close_draw_odds",
        "avg_close_away_odds",
    ]

    for column in (
        numeric_columns
    ):

        if (
            column
            in combined.columns
        ):

            combined[
                column
            ] = pd.to_numeric(
                combined[
                    column
                ],
                errors="coerce"
            )

    # ========================================================
    # SORT
    # ========================================================

    combined = (
        combined
        .dropna(
            subset=[
                "date",
                "league_code",
                "home_team",
                "away_team",
            ]
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

    return (
        combined,
        len(
            new_keys
        ),
        len(
            common_keys
        ),
    )


# ============================================================
# REFRESH CURRENT SEASON
# ============================================================

def refresh_current_season(
    matches,
    save=False,
    verbose=True
):

    season = (
        get_current_season_code()
    )

    matches = (
        matches.copy()
    )

    matches[
        "date"
    ] = pd.to_datetime(
        matches[
            "date"
        ],
        errors="coerce"
    )

    template_columns = list(
        matches.columns
    )

    combined = (
        matches.copy()
    )

    report = {

        "season":
            season,

        "new_matches":
            0,

        "refreshed_matches":
            0,

        "league_results":
            {},

        "errors":
            {},
    }

    if verbose:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "UPDATING CURRENT SEASON "
            f"{season[:2]}/{season[2:]}"
        )

        print(
            "=" * 72
        )

    # ========================================================
    # EACH BIG FIVE LEAGUE
    # ========================================================

    for league_code in (
        LEAGUES
    ):

        league_name = (
            LEAGUES[
                league_code
            ]
        )

        if verbose:

            print(
                f"\n{league_name}"
            )

        try:

            raw = (
                download_league(
                    league_code,
                    season
                )
            )

            current_rows = (
                build_current_rows(
                    raw,
                    league_code,
                    season,
                    template_columns
                )
            )

            (
                combined,
                new_count,
                refreshed_count,
            ) = merge_matches(
                combined,
                current_rows
            )

            report[
                "new_matches"
            ] += (
                new_count
            )

            report[
                "refreshed_matches"
            ] += (
                refreshed_count
            )

            report[
                "league_results"
            ][
                league_name
            ] = {

                "completed":
                    len(
                        current_rows
                    ),

                "new":
                    new_count,

                "refreshed":
                    refreshed_count,
            }

            if verbose:

                print(
                    "  Completed in source: "
                    f"{len(current_rows)}"
                )

                print(
                    "  New matches: "
                    f"{new_count}"
                )

                print(
                    "  Existing refreshed: "
                    f"{refreshed_count}"
                )

        except Exception as error:

            report[
                "errors"
            ][
                league_name
            ] = str(
                error
            )

            if verbose:

                print(
                    f"  ERROR: "
                    f"{error}"
                )

    # ========================================================
    # SAVE DATABASE
    # ========================================================

    if save:

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # BACKUP ORIGINAL
        # ----------------------------------------------------

        if MATCHES_FILE.exists():

            matches.to_csv(
                BACKUP_FILE,
                index=False
            )

        # ----------------------------------------------------
        # SAVE UPDATED DATABASE
        # ----------------------------------------------------

        combined.to_csv(
            MATCHES_FILE,
            index=False
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    latest_date = (
        combined[
            "date"
        ].max()
    )

    if verbose:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "UPDATE COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            f"Matches before: "
            f"{len(matches):,}"
        )

        print(
            f"Matches after:  "
            f"{len(combined):,}"
        )

        print(
            f"New matches:    "
            f"{report['new_matches']}"
        )

        print(
            f"Refreshed:      "
            f"{report['refreshed_matches']}"
        )

        if pd.notna(
            latest_date
        ):

            print(
                f"Latest date:    "
                f"{latest_date.date()}"
            )

        if report[
            "errors"
        ]:

            print(
                "\nWARNING:"
            )

            print(
                "Some leagues failed to update:"
            )

            for (
                league_name,
                error,
            ) in (
                report[
                    "errors"
                ].items()
            ):

                print(
                    f"  {league_name}: "
                    f"{error}"
                )

        else:

            print(
                "\nAll five leagues "
                "updated successfully."
            )

        if save:

            print(
                "\nDatabase:"
            )

            print(
                MATCHES_FILE
            )

            print(
                "\nBackup:"
            )

            print(
                BACKUP_FILE
            )

    return (
        combined,
        report,
    )


# ============================================================
# TERMINAL RUN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LIVE MATCH DATABASE UPDATER"
    )

    print(
        "=" * 72
    )

    if not MATCHES_FILE.exists():

        print(
            "\nERROR:"
        )

        print(
            "Could not find database:"
        )

        print(
            MATCHES_FILE
        )

        return

    # ========================================================
    # LOAD DATABASE
    # ========================================================

    print(
        "\nLoading current database..."
    )

    matches = pd.read_csv(
        MATCHES_FILE,
        low_memory=False
    )

    matches[
        "date"
    ] = pd.to_datetime(
        matches[
            "date"
        ],
        errors="coerce"
    )

    print(
        f"Matches currently stored: "
        f"{len(matches):,}"
    )

    latest_before = (
        matches[
            "date"
        ].max()
    )

    if pd.notna(
        latest_before
    ):

        print(
            f"Latest match currently stored: "
            f"{latest_before.date()}"
        )

    # ========================================================
    # UPDATE
    # ========================================================

    refresh_current_season(
        matches,
        save=True,
        verbose=True
    )


if __name__ == "__main__":
    run()
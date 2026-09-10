import sys
import re
import math
import difflib
import unicodedata

import requests
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, LEAGUES

from .match_predictor import (
    load_matches,
    predict_match,
)

from .match_stats_predictor import (
    resolve_stat_columns,
    predict_metric,
)


# ============================================================
# SETTINGS
# ============================================================

REQUEST_TIMEOUT = 30
MAX_SCORE_GOALS = 8


# ESPN league IDs
ESPN_LEAGUES = {
    "E0": "eng.1",
    "SP1": "esp.1",
    "I1": "ita.1",
    "D1": "ger.1",
    "F1": "fra.1",
}


# ============================================================
# TEAM NAME ALIASES
#
# ESPN and football-data.co.uk sometimes use different names.
# ============================================================

TEAM_ALIASES = {

    # --------------------------------------------------------
    # ENGLAND
    # --------------------------------------------------------

    "manchester united":
        "Man United",

    "manchester city":
        "Man City",

    "tottenham hotspur":
        "Tottenham",

    "newcastle united":
        "Newcastle",

    "west ham united":
        "West Ham",

    "wolverhampton wanderers":
        "Wolves",

    "nottingham forest":
        "Nott'm Forest",

    "brighton hove albion":
        "Brighton",

    "brighton and hove albion":
        "Brighton",

    "afc bournemouth":
        "Bournemouth",

    "leeds united":
        "Leeds",

    "ipswich town":
        "Ipswich",

    "leicester city":
        "Leicester",

    # --------------------------------------------------------
    # SPAIN
    # --------------------------------------------------------

    "fc barcelona":
        "Barcelona",

    "atletico madrid":
        "Ath Madrid",

    "atletico de madrid":
        "Ath Madrid",

    "athletic club":
        "Ath Bilbao",

    "athletic bilbao":
        "Ath Bilbao",

    "real sociedad":
        "Sociedad",

    "real betis":
        "Betis",

    "real betis balompie":
        "Betis",

    "sevilla fc":
        "Sevilla",

    "valencia cf":
        "Valencia",

    "villarreal cf":
        "Villarreal",

    "getafe cf":
        "Getafe",

    "celta vigo":
        "Celta",

    "rc celta":
        "Celta",

    "rc celta de vigo":
        "Celta",

    "deportivo alaves":
        "Alaves",

    "ca osasuna":
        "Osasuna",

    "rcd espanyol":
        "Espanol",

    "rayo vallecano":
        "Vallecano",

    "levante ud":
        "Levante",

    "elche cf":
        "Elche",

    "rcd mallorca":
        "Mallorca",

    "real mallorca":
        "Mallorca",

    "girona fc":
        "Girona",

    # --------------------------------------------------------
    # ITALY
    # --------------------------------------------------------

    "internazionale":
        "Inter",

    "inter milan":
        "Inter",

    "ac milan":
        "Milan",

    "as roma":
        "Roma",

    "ssc napoli":
        "Napoli",

    "ss lazio":
        "Lazio",

    "hellas verona":
        "Verona",

    # --------------------------------------------------------
    # GERMANY
    # --------------------------------------------------------

    "bayern munich":
        "Bayern Munich",

    "bayern munchen":
        "Bayern Munich",

    "borussia dortmund":
        "Dortmund",

    "bayer leverkusen":
        "Leverkusen",

    "bayer 04 leverkusen":
        "Leverkusen",

    "eintracht frankfurt":
        "Ein Frankfurt",

    "borussia monchengladbach":
        "M'gladbach",

    "rb leipzig":
        "RB Leipzig",

    "werder bremen":
        "Werder Bremen",

    "vfb stuttgart":
        "Stuttgart",

    "sc freiburg":
        "Freiburg",

    "union berlin":
        "Union Berlin",

    "1 fc union berlin":
        "Union Berlin",

    "fc augsburg":
        "Augsburg",

    "hamburger sv":
        "Hamburg",

    "fc cologne":
        "FC Koln",

    "1 fc koln":
        "FC Koln",

    "cologne":
        "FC Koln",

    # --------------------------------------------------------
    # FRANCE
    # --------------------------------------------------------

    "paris saint germain":
        "Paris SG",

    "olympique marseille":
        "Marseille",

    "olympique de marseille":
        "Marseille",

    "olympique lyonnais":
        "Lyon",

    "as monaco":
        "Monaco",

    "losc lille":
        "Lille",

    "rc lens":
        "Lens",

    "stade rennais":
        "Rennes",

    "stade rennais fc":
        "Rennes",

    "ogc nice":
        "Nice",

    "rc strasbourg":
        "Strasbourg",
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_team_name(name):

    if name is None:
        return ""

    text = str(name).lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIMPLIFIED TEAM NAME
# ============================================================

def simplified_team_name(name):

    text = normalize_team_name(
        name
    )

    ignore_words = {
        "fc",
        "cf",
        "afc",
        "sc",
        "ss",
        "ssc",
        "rc",
        "rcd",
        "ud",
        "sv",
        "vfb",
        "as",
        "ogc",
        "losc",
        "club",
        "football",
        "futbol",
        "calcio",
        "de",
        "the",
    }

    words = [
        word
        for word in text.split()
        if word not in ignore_words
    ]

    return " ".join(
        words
    )


# ============================================================
# GET TEAMS AVAILABLE BEFORE DATE
# ============================================================

def available_teams_for_date(
    matches,
    league_code,
    prediction_date
):

    league = matches[
        (
            matches[
                "league_code"
            ]
            == league_code
        )
        &
        (
            matches[
                "date"
            ]
            < prediction_date
        )
    ].copy()

    if len(league) == 0:
        return []

    # First prefer teams active relatively recently.
    cutoff = (
        prediction_date
        - pd.Timedelta(
            days=550
        )
    )

    recent = league[
        league[
            "date"
        ]
        >= cutoff
    ]

    source = (
        recent
        if len(recent) > 0
        else league
    )

    teams = set(
        source[
            "home_team"
        ]
        .dropna()
        .astype(str)
    )

    teams.update(
        source[
            "away_team"
        ]
        .dropna()
        .astype(str)
    )

    return sorted(
        teams
    )


# ============================================================
# MATCH EXTERNAL TEAM NAME TO DATASET TEAM NAME
# ============================================================

def match_team_name(
    external_name,
    teams
):

    if not teams:

        return (
            None,
            0.0
        )

    normalized_external = (
        normalize_team_name(
            external_name
        )
    )

    simplified_external = (
        simplified_team_name(
            external_name
        )
    )

    # ========================================================
    # 1. MANUAL ALIAS
    # ========================================================

    if (
        normalized_external
        in TEAM_ALIASES
    ):

        alias = (
            TEAM_ALIASES[
                normalized_external
            ]
        )

        normalized_alias = (
            normalize_team_name(
                alias
            )
        )

        for team in teams:

            if (
                normalize_team_name(
                    team
                )
                ==
                normalized_alias
            ):

                return (
                    team,
                    1.0
                )

    # ========================================================
    # 2. EXACT NORMALIZED MATCH
    # ========================================================

    for team in teams:

        if (
            normalize_team_name(
                team
            )
            ==
            normalized_external
        ):

            return (
                team,
                1.0
            )

    # ========================================================
    # 3. EXACT SIMPLIFIED MATCH
    # ========================================================

    for team in teams:

        if (
            simplified_team_name(
                team
            )
            ==
            simplified_external
        ):

            return (
                team,
                0.99
            )

    # ========================================================
    # 4. ONE NAME CONTAINS THE OTHER
    # ========================================================

    for team in teams:

        candidate = (
            simplified_team_name(
                team
            )
        )

        if (
            candidate
            and
            simplified_external
            and
            (
                candidate
                in simplified_external
                or
                simplified_external
                in candidate
            )
        ):

            return (
                team,
                0.92
            )

    # ========================================================
    # 5. FUZZY MATCH
    # ========================================================

    best_team = None
    best_score = 0.0

    external_words = set(
        simplified_external.split()
    )

    for team in teams:

        candidate = (
            simplified_team_name(
                team
            )
        )

        sequence_score = (
            difflib.SequenceMatcher(
                None,
                simplified_external,
                candidate
            ).ratio()
        )

        candidate_words = set(
            candidate.split()
        )

        if (
            external_words
            and candidate_words
        ):

            overlap_score = (
                len(
                    external_words
                    &
                    candidate_words
                )
                /
                len(
                    external_words
                    |
                    candidate_words
                )
            )

        else:

            overlap_score = 0.0

        score = max(
            sequence_score,
            overlap_score
        )

        if score > best_score:

            best_score = score
            best_team = team

    # Avoid dangerous weak matches.
    if (
        best_team is not None
        and best_score >= 0.70
    ):

        return (
            best_team,
            best_score
        )

    return (
        None,
        best_score
    )


# ============================================================
# ESPN FIXTURE DOWNLOAD
# ============================================================

def get_league_fixtures(
    league_code,
    prediction_date
):

    espn_code = (
        ESPN_LEAGUES[
            league_code
        ]
    )

    date_string = (
        pd.Timestamp(
            prediction_date
        ).strftime(
            "%Y%m%d"
        )
    )

    url = (
        "https://site.api.espn.com/"
        "apis/site/v2/sports/soccer/"
        f"{espn_code}/scoreboard"
    )

    print(
        f"\nChecking "
        f"{LEAGUES[league_code]} "
        f"for {date_string}..."
    )

    try:

        response = requests.get(
            url,
            params={
                "dates":
                    date_string,

                "limit":
                    100,
            },
            timeout=
                REQUEST_TIMEOUT,
        )

        print(
            f"HTTP status: "
            f"{response.status_code}"
        )

        response.raise_for_status()

        data = (
            response.json()
        )

    except requests.RequestException as error:

        print(
            f"ERROR retrieving "
            f"{LEAGUES[league_code]}:"
        )

        print(
            error
        )

        return []

    except ValueError as error:

        print(
            f"ERROR reading ESPN response "
            f"for {LEAGUES[league_code]}:"
        )

        print(
            error
        )

        return []

    events = (
        data.get(
            "events"
        )
        or []
    )

    print(
        f"Raw events found: "
        f"{len(events)}"
    )

    fixtures = []

    seen = set()

    for event in events:

        competitions = (
            event.get(
                "competitions"
            )
            or []
        )

        if not competitions:
            continue

        competition = (
            competitions[0]
        )

        competitors = (
            competition.get(
                "competitors"
            )
            or []
        )

        home_team = None
        away_team = None

        for competitor in competitors:

            team_data = (
                competitor.get(
                    "team"
                )
                or {}
            )

            team_name = (
                team_data.get(
                    "displayName"
                )
                or
                team_data.get(
                    "shortDisplayName"
                )
                or
                team_data.get(
                    "name"
                )
            )

            side = (
                competitor.get(
                    "homeAway"
                )
            )

            if side == "home":

                home_team = (
                    team_name
                )

            elif side == "away":

                away_team = (
                    team_name
                )

        if (
            not home_team
            or not away_team
        ):
            continue

        event_id = (
            event.get(
                "id"
            )
        )

        duplicate_key = (
            event_id
            or
            (
                home_team,
                away_team,
                event.get(
                    "date"
                ),
            )
        )

        if duplicate_key in seen:
            continue

        seen.add(
            duplicate_key
        )

        status_data = (
            event.get(
                "status",
                {}
            )
            .get(
                "type",
                {}
            )
        )

        status = (
            status_data.get(
                "detail"
            )
            or
            status_data.get(
                "description"
            )
            or
            status_data.get(
                "name"
            )
            or
            "Unknown"
        )

        fixture = {
            "event_id":
                event_id,

            "league_code":
                league_code,

            "league":
                LEAGUES[
                    league_code
                ],

            "home_external":
                home_team,

            "away_external":
                away_team,

            "kickoff":
                event.get(
                    "date"
                ),

            "status":
                status,
        }

        fixtures.append(
            fixture
        )

        print(
            f"  FOUND: "
            f"{home_team} "
            f"vs "
            f"{away_team}"
        )

    print(
        f"Usable fixtures: "
        f"{len(fixtures)}"
    )

    return fixtures


# ============================================================
# ALL BIG FIVE FIXTURES
# ============================================================

def get_all_fixtures(
    prediction_date
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DOWNLOADING BIG FIVE FIXTURES"
    )

    print(
        "=" * 72
    )

    all_fixtures = []

    for league_code in (
        ESPN_LEAGUES
    ):

        fixtures = (
            get_league_fixtures(
                league_code,
                prediction_date
            )
        )

        all_fixtures.extend(
            fixtures
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"TOTAL FIXTURES FOUND: "
        f"{len(all_fixtures)}"
    )

    print(
        "=" * 72
    )

    return all_fixtures


# ============================================================
# POISSON SCORE PROBABILITY
# ============================================================

def poisson_probability(
    goals,
    expected_goals
):

    expected_goals = max(
        float(
            expected_goals
        ),
        0.001
    )

    return (
        math.exp(
            -expected_goals
        )
        *
        (
            expected_goals
            ** goals
        )
        /
        math.factorial(
            goals
        )
    )


# ============================================================
# MOST LIKELY EXACT SCORE
# ============================================================

def most_likely_score(
    home_xg,
    away_xg
):

    best_home = 0
    best_away = 0
    best_probability = -1.0

    for home_goals in range(
        MAX_SCORE_GOALS + 1
    ):

        home_probability = (
            poisson_probability(
                home_goals,
                home_xg
            )
        )

        for away_goals in range(
            MAX_SCORE_GOALS + 1
        ):

            probability = (
                home_probability
                *
                poisson_probability(
                    away_goals,
                    away_xg
                )
            )

            if (
                probability
                >
                best_probability
            ):

                best_probability = (
                    probability
                )

                best_home = (
                    home_goals
                )

                best_away = (
                    away_goals
                )

    return (
        best_home,
        best_away,
        best_probability,
    )


# ============================================================
# RESULT LABEL
# ============================================================

def get_result_name(
    best_result,
    home_team,
    away_team
):

    if best_result == "H":

        return (
            f"{home_team} WIN"
        )

    if best_result == "A":

        return (
            f"{away_team} WIN"
        )

    return "DRAW"


# ============================================================
# SIMULATE ONE FIXTURE
# ============================================================

def simulate_fixture(
    fixture,
    matches,
    stat_columns,
    prediction_date
):

    league_code = (
        fixture[
            "league_code"
        ]
    )

    teams = (
        available_teams_for_date(
            matches,
            league_code,
            prediction_date
        )
    )

    (
        home_team,
        home_match_quality,
    ) = match_team_name(
        fixture[
            "home_external"
        ],
        teams
    )

    (
        away_team,
        away_match_quality,
    ) = match_team_name(
        fixture[
            "away_external"
        ],
        teams
    )

    # ========================================================
    # TEAM MATCHING FAILURE
    # ========================================================

    if (
        home_team is None
        or away_team is None
    ):

        print(
            "\nTEAM MATCHING FAILED"
        )

        print(
            f"  HOME: "
            f"{fixture['home_external']} "
            f"-> {home_team}"
        )

        print(
            f"  AWAY: "
            f"{fixture['away_external']} "
            f"-> {away_team}"
        )

        return None

    if (
        home_match_quality < 0.85
        or away_match_quality < 0.85
    ):

        print(
            "\nTeam-name mapping:"
        )

        print(
            f"  "
            f"{fixture['home_external']} "
            f"-> "
            f"{home_team} "
            f"({home_match_quality:.0%})"
        )

        print(
            f"  "
            f"{fixture['away_external']} "
            f"-> "
            f"{away_team} "
            f"({away_match_quality:.0%})"
        )

    # ========================================================
    # RESULT + EXPECTED GOALS
    # ========================================================

    football = (
        predict_match(
            matches,
            home_team,
            away_team,
            league_code,
            prediction_date
        )
    )

    probabilities = np.asarray(
        football[
            "probabilities"
        ],
        dtype=float
    )

    if (
        len(probabilities)
        != 3
    ):

        raise ValueError(
            "predict_match() did not return "
            "three H/D/A probabilities."
        )

    home_xg = float(
        football[
            "home_xg"
        ]
    )

    away_xg = float(
        football[
            "away_xg"
        ]
    )

    (
        predicted_home_goals,
        predicted_away_goals,
        score_probability,
    ) = most_likely_score(
        home_xg,
        away_xg
    )

    result_codes = [
        "H",
        "D",
        "A",
    ]

    best_result = (
        result_codes[
            int(
                np.argmax(
                    probabilities
                )
            )
        ]
    )

    # ========================================================
    # SHOTS ON TARGET
    # ========================================================

    (
        home_sot,
        away_sot,
    ) = predict_metric(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date,
        stat_columns[
            "home_sot"
        ],
        stat_columns[
            "away_sot"
        ]
    )

    # ========================================================
    # CORNERS
    # ========================================================

    (
        home_corners,
        away_corners,
    ) = predict_metric(
        matches,
        home_team,
        away_team,
        league_code,
        prediction_date,
        stat_columns[
            "home_corners"
        ],
        stat_columns[
            "away_corners"
        ]
    )

    return {

        # ----------------------------------------------------
        # FIXTURE
        # ----------------------------------------------------

        "date":
            prediction_date.date(),

        "league_code":
            league_code,

        "league":
            fixture[
                "league"
            ],

        "event_id":
            fixture[
                "event_id"
            ],

        "kickoff":
            fixture[
                "kickoff"
            ],

        "status":
            fixture[
                "status"
            ],

        "home_team":
            home_team,

        "away_team":
            away_team,

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        "home_probability":
            float(
                probabilities[0]
            ),

        "draw_probability":
            float(
                probabilities[1]
            ),

        "away_probability":
            float(
                probabilities[2]
            ),

        "predicted_result":
            get_result_name(
                best_result,
                home_team,
                away_team
            ),

        # ----------------------------------------------------
        # GOALS
        # ----------------------------------------------------

        "home_xg":
            home_xg,

        "away_xg":
            away_xg,

        "total_xg":
            home_xg
            + away_xg,

        "predicted_home_goals":
            predicted_home_goals,

        "predicted_away_goals":
            predicted_away_goals,

        "predicted_score":
            (
                f"{predicted_home_goals}"
                "-"
                f"{predicted_away_goals}"
            ),

        "score_probability":
            float(
                score_probability
            ),

        # ----------------------------------------------------
        # SHOTS ON TARGET
        # ----------------------------------------------------

        "home_sot":
            float(
                home_sot
            ),

        "away_sot":
            float(
                away_sot
            ),

        "total_sot":
            float(
                home_sot
                + away_sot
            ),

        # ----------------------------------------------------
        # CORNERS
        # ----------------------------------------------------

        "home_corners":
            float(
                home_corners
            ),

        "away_corners":
            float(
                away_corners
            ),

        "total_corners":
            float(
                home_corners
                + away_corners
            ),
    }


# ============================================================
# DISPLAY ONE MATCH
# ============================================================

def display_match(
    prediction
):

    home = (
        prediction[
            "home_team"
        ]
    )

    away = (
        prediction[
            "away_team"
        ]
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        f"{home} vs {away}"
    )

    print(
        f"{prediction['league']} "
        f"| "
        f"{prediction['status']}"
    )

    print(
        "=" * 76
    )

    # ========================================================
    # RESULT
    # ========================================================

    print(
        "\nRESULT"
    )

    print(
        "-" * 48
    )

    print(
        f"{home:<30}"
        f"{prediction['home_probability']:>10.1%}"
    )

    print(
        f"{'DRAW':<30}"
        f"{prediction['draw_probability']:>10.1%}"
    )

    print(
        f"{away:<30}"
        f"{prediction['away_probability']:>10.1%}"
    )

    print(
        f"\nPrediction: "
        f"{prediction['predicted_result']}"
    )

    # ========================================================
    # GOALS
    # ========================================================

    print(
        "\nGOALS"
    )

    print(
        "-" * 48
    )

    print(
        f"{home:<30}"
        f"{prediction['home_xg']:>8.2f} xG"
    )

    print(
        f"{away:<30}"
        f"{prediction['away_xg']:>8.2f} xG"
    )

    print(
        f"{'TOTAL':<30}"
        f"{prediction['total_xg']:>8.2f} xG"
    )

    print(
        f"\nMost likely score: "
        f"{home} "
        f"{prediction['predicted_home_goals']}"
        " - "
        f"{prediction['predicted_away_goals']} "
        f"{away}"
    )

    print(
        "Exact-score probability: "
        f"{prediction['score_probability']:.1%}"
    )

    # ========================================================
    # SHOTS ON TARGET
    # ========================================================

    print(
        "\nSHOTS ON TARGET"
    )

    print(
        "-" * 48
    )

    print(
        f"{home:<30}"
        f"{prediction['home_sot']:>8.2f}"
    )

    print(
        f"{away:<30}"
        f"{prediction['away_sot']:>8.2f}"
    )

    print(
        f"{'TOTAL':<30}"
        f"{prediction['total_sot']:>8.2f}"
    )

    # ========================================================
    # CORNERS
    # ========================================================

    print(
        "\nCORNERS"
    )

    print(
        "-" * 48
    )

    print(
        f"{home:<30}"
        f"{prediction['home_corners']:>8.2f}"
    )

    print(
        f"{away:<30}"
        f"{prediction['away_corners']:>8.2f}"
    )

    print(
        f"{'TOTAL':<30}"
        f"{prediction['total_corners']:>8.2f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def display_summary(
    predictions
):

    print(
        "\n"
        + "=" * 138
    )

    print(
        "DAILY PREDICTION SUMMARY"
    )

    print(
        "=" * 138
    )

    print(
        f"{'LEAGUE':<18}"
        f"{'MATCH':<36}"
        f"{'PREDICTION':<23}"
        f"{'SCORE':>8}"
        f"{'xG':>12}"
        f"{'SOT':>12}"
        f"{'CORNERS':>12}"
    )

    print(
        "-" * 138
    )

    for row in predictions:

        match_name = (
            f"{row['home_team']} "
            f"vs "
            f"{row['away_team']}"
        )

        # Prevent very long club names from breaking table.
        if len(match_name) > 35:

            match_name = (
                match_name[:32]
                + "..."
            )

        prediction_name = (
            row[
                "predicted_result"
            ]
        )

        if len(prediction_name) > 22:

            prediction_name = (
                prediction_name[:19]
                + "..."
            )

        xg = (
            f"{row['home_xg']:.1f}"
            "-"
            f"{row['away_xg']:.1f}"
        )

        sot = (
            f"{row['home_sot']:.1f}"
            "-"
            f"{row['away_sot']:.1f}"
        )

        corners = (
            f"{row['home_corners']:.1f}"
            "-"
            f"{row['away_corners']:.1f}"
        )

        print(
            f"{row['league']:<18}"
            f"{match_name:<36}"
            f"{prediction_name:<23}"
            f"{row['predicted_score']:>8}"
            f"{xg:>12}"
            f"{sot:>12}"
            f"{corners:>12}"
        )


# ============================================================
# DISPLAY FAILED FIXTURES
# ============================================================

def display_failed_fixtures(
    failed
):

    if not failed:
        return

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FIXTURES THAT COULD NOT BE PREDICTED"
    )

    print(
        "=" * 76
    )

    for fixture in failed:

        print(
            f"{fixture['league']}: "
            f"{fixture['home_external']} "
            f"vs "
            f"{fixture['away_external']}"
        )


# ============================================================
# RUN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 76
    )

    print(
        "BIG FIVE DAILY MATCH SIMULATOR"
    )

    print(
        "=" * 76
    )

    print(
        "Result + Goals + xG + "
        "Shots on Target + Corners"
    )

    # ========================================================
    # DATE
    # ========================================================

    if len(
        sys.argv
    ) >= 2:

        date_value = (
            sys.argv[1]
            .strip()
        )

    else:

        date_value = input(
            "\nDate YYYY-MM-DD "
            "(Enter for today): "
        ).strip()

    if date_value == "":

        prediction_date = (
            pd.Timestamp.today()
            .normalize()
        )

    else:

        try:

            prediction_date = (
                pd.Timestamp(
                    date_value
                ).normalize()
            )

        except Exception:

            print(
                "\nInvalid date."
            )

            print(
                "Use format YYYY-MM-DD."
            )

            return

    print(
        f"\nSimulation date: "
        f"{prediction_date.date()}"
    )

    # ========================================================
    # HISTORICAL DATA
    # ========================================================

    print(
        "\nLoading historical data..."
    )

    matches = (
        load_matches()
    )

    matches[
        "date"
    ] = pd.to_datetime(
        matches[
            "date"
        ],
        errors="coerce"
    )

    matches = (
        matches
        .dropna(
            subset=[
                "date",
                "league_code",
                "home_team",
                "away_team",
            ]
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    stat_columns = (
        resolve_stat_columns(
            matches
        )
    )

    print(
        f"Historical matches: "
        f"{len(matches):,}"
    )

    print(
        f"SOT columns: "
        f"{stat_columns['home_sot']} / "
        f"{stat_columns['away_sot']}"
    )

    print(
        f"Corner columns: "
        f"{stat_columns['home_corners']} / "
        f"{stat_columns['away_corners']}"
    )

    # ========================================================
    # FIXTURES
    # ========================================================

    fixtures = (
        get_all_fixtures(
            prediction_date
        )
    )

    if not fixtures:

        print(
            "\nNo Big Five games were "
            "returned for that date."
        )

        print(
            "\nTry a known matchday date."
        )

        return

    # ========================================================
    # SHOW RAW FIXTURE LIST
    # ========================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "FIXTURES FOUND"
    )

    print(
        "=" * 76
    )

    for number, fixture in enumerate(
        fixtures,
        start=1
    ):

        print(
            f"{number:>2}. "
            f"{fixture['league']:<18} "
            f"{fixture['home_external']} "
            f"vs "
            f"{fixture['away_external']}"
        )

    # ========================================================
    # SIMULATIONS
    # ========================================================

    predictions = []
    failed = []

    print(
        "\n"
        + "=" * 76
    )

    print(
        "RUNNING PREDICTIONS"
    )

    print(
        "=" * 76
    )

    for number, fixture in enumerate(
        fixtures,
        start=1
    ):

        print(
            f"\n[{number}/{len(fixtures)}] "
            f"{fixture['home_external']} "
            f"vs "
            f"{fixture['away_external']}"
        )

        try:

            prediction = (
                simulate_fixture(
                    fixture,
                    matches,
                    stat_columns,
                    prediction_date
                )
            )

        except Exception as error:

            print(
                f"\nPrediction error: "
                f"{error}"
            )

            prediction = None

        if prediction is None:

            failed.append(
                fixture
            )

            continue

        predictions.append(
            prediction
        )

        display_match(
            prediction
        )

    # ========================================================
    # RESULTS
    # ========================================================

    if predictions:

        display_summary(
            predictions
        )

    display_failed_fixtures(
        failed
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if predictions:

        output = (
            pd.DataFrame(
                predictions
            )
        )

        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            "daily_match_stats_"
            f"{prediction_date.date()}"
            ".csv"
        )

        output_path = (
            PROCESSED_DIR
            /
            filename
        )

        output.to_csv(
            output_path,
            index=False
        )

        print(
            "\n"
            + "=" * 76
        )

        print(
            "SIMULATION COMPLETE"
        )

        print(
            "=" * 76
        )

        print(
            f"Fixtures found:       "
            f"{len(fixtures)}"
        )

        print(
            f"Successfully modeled: "
            f"{len(predictions)}"
        )

        print(
            f"Failed mappings:      "
            f"{len(failed)}"
        )

        print(
            f"Saved: "
            f"{output_path}"
        )

    else:

        print(
            "\nNo fixtures could be "
            "successfully modeled."
        )


if __name__ == "__main__":
    run()
import json
import re
from datetime import datetime, timezone

import requests


# ============================================================
# SETTINGS
# ============================================================

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"

REQUEST_TIMEOUT = 20

SEARCH_PAGES = 3
SEARCH_LIMIT = 50


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# TEAM ALIASES
# ============================================================

TEAM_ALIASES = {

    "man utd":
        "manchester united",

    "man united":
        "manchester united",

    "man city":
        "manchester city",

    "spurs":
        "tottenham hotspur",

    "tottenham":
        "tottenham hotspur",

    "barca":
        "barcelona",

    "psg":
        "paris saint germain",

    "inter milan":
        "inter",

    "ac milan":
        "milan",

    "bayern munich":
        "bayern",

    "atletico madrid":
        "atletico madrid",
}


IGNORE_WORDS = {
    "fc",
    "cf",
    "afc",
    "club",
    "de",
    "calcio",
    "1901",
}


def canonical_team_name(
    name
):

    name = normalize_text(
        name
    )

    return TEAM_ALIASES.get(
        name,
        name
    )


def team_tokens(
    name
):

    canonical = canonical_team_name(
        name
    )

    return [
        token
        for token in canonical.split()
        if token not in IGNORE_WORDS
    ]


# ============================================================
# REQUEST
# ============================================================

def get_json(
    url,
    params=None
):

    try:

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent":
                    "soccer-prediction-phase3/1.0"
            }
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        print(
            "\nAPI ERROR:"
        )

        print(
            error
        )

        return None

    except ValueError:

        print(
            "\nInvalid JSON response."
        )

        return None


# ============================================================
# PARSE GAMMA ARRAY
# ============================================================

def parse_array(
    value
):

    if value is None:
        return []

    if isinstance(
        value,
        list
    ):
        return value

    if isinstance(
        value,
        tuple
    ):
        return list(value)

    if isinstance(
        value,
        str
    ):

        try:

            result = json.loads(
                value
            )

            if isinstance(
                result,
                list
            ):
                return result

        except json.JSONDecodeError:
            pass

    return []


# ============================================================
# PUBLIC SEARCH
#
# Search is used ONLY to discover candidates.
# We DO NOT trust the search result by itself.
# ============================================================

def public_search(
    query
):

    found = {}

    for page in range(
        1,
        SEARCH_PAGES + 1
    ):

        data = get_json(
            f"{GAMMA_BASE_URL}/public-search",
            params={
                "q":
                    query,

                "events_status":
                    "active",

                "keep_closed_markets":
                    0,

                "limit_per_type":
                    SEARCH_LIMIT,

                "page":
                    page,

                "search_tags":
                    "false",

                "search_profiles":
                    "false",
            }
        )

        if not isinstance(
            data,
            dict
        ):
            break

        events = (
            data.get(
                "events"
            )
            or []
        )

        for event in events:

            if not isinstance(
                event,
                dict
            ):
                continue

            event_id = str(
                event.get(
                    "id",
                    ""
                )
            )

            if event_id:

                found[
                    event_id
                ] = event

        pagination = (
            data.get(
                "pagination"
            )
            or {}
        )

        if not pagination.get(
            "hasMore",
            False
        ):
            break

    return list(
        found.values()
    )


# ============================================================
# FETCH FULL EVENT
# ============================================================

def get_event(
    event_id
):

    return get_json(
        f"{GAMMA_BASE_URL}/events/{event_id}"
    )


# ============================================================
# STRUCTURED SPORTS DATA
# ============================================================

def get_sports_object(
    event
):

    sports = event.get(
        "sports"
    )

    if isinstance(
        sports,
        dict
    ):
        return sports

    return {}


def get_event_teams(
    event
):

    # Gamma responses may expose teams directly
    # or under the sports object.

    teams = event.get(
        "teams"
    )

    if isinstance(
        teams,
        list
    ) and teams:

        return teams

    sports = get_sports_object(
        event
    )

    teams = sports.get(
        "teams"
    )

    if isinstance(
        teams,
        list
    ):

        return teams

    return []


def get_game_id(
    event
):

    game_id = event.get(
        "gameId"
    )

    if game_id is not None:
        return game_id

    sports = get_sports_object(
        event
    )

    return sports.get(
        "gameId"
    )


# ============================================================
# TEAM MATCHING
# ============================================================

def team_object_matches(
    typed_team,
    team_object
):

    if not isinstance(
        team_object,
        dict
    ):
        return False

    typed = canonical_team_name(
        typed_team
    )

    typed_tokens = set(
        team_tokens(
            typed_team
        )
    )

    possible_names = [
        team_object.get(
            "name"
        ),
        team_object.get(
            "abbreviation"
        ),
    ]

    for possible in possible_names:

        if not possible:
            continue

        possible = canonical_team_name(
            possible
        )

        # Exact
        if possible == typed:
            return True

        # Chelsea -> Chelsea FC
        if (
            typed in possible
            or possible in typed
        ):
            return True

        possible_tokens = set(
            team_tokens(
                possible
            )
        )

        if (
            typed_tokens
            and
            typed_tokens.issubset(
                possible_tokens
            )
        ):
            return True

    return False


# ============================================================
# STRICT MATCH CHECK
#
# This is the important fix.
#
# We require an actual sports event with a structured HOME
# team and AWAY team.
#
# Futures markets such as:
#
# "Will Arsenal win the Premier League?"
#
# will fail this test.
# ============================================================

def is_exact_game_event(
    event,
    home_team,
    away_team
):

    teams = get_event_teams(
        event
    )

    if not teams:
        return False

    structured_home = None
    structured_away = None

    for team in teams:

        if not isinstance(
            team,
            dict
        ):
            continue

        ordering = normalize_text(
            team.get(
                "ordering"
            )
        )

        if ordering == "home":

            structured_home = team

        elif ordering == "away":

            structured_away = team

    # Real sports game events should identify
    # home and away teams.
    if (
        structured_home is None
        or structured_away is None
    ):
        return False

    correct_home = (
        team_object_matches(
            home_team,
            structured_home
        )
    )

    correct_away = (
        team_object_matches(
            away_team,
            structured_away
        )
    )

    return (
        correct_home
        and correct_away
    )


# ============================================================
# EVENT DATE
# ============================================================

def parse_event_datetime(
    event
):

    sports = get_sports_object(
        event
    )

    possible = [

        event.get(
            "startTime"
        ),

        event.get(
            "eventDate"
        ),

        event.get(
            "startDate"
        ),

        sports.get(
            "startTime"
        ),

        sports.get(
            "eventDate"
        ),
    ]

    for value in possible:

        if not value:
            continue

        try:

            text = str(
                value
            ).replace(
                "Z",
                "+00:00"
            )

            result = datetime.fromisoformat(
                text
            )

            if result.tzinfo is None:

                result = result.replace(
                    tzinfo=timezone.utc
                )

            return result

        except ValueError:
            continue

    return None


# ============================================================
# DATE FILTER
# ============================================================

def matches_requested_date(
    event,
    requested_date
):

    if requested_date is None:
        return True

    event_datetime = (
        parse_event_datetime(
            event
        )
    )

    if event_datetime is None:
        return False

    return (
        event_datetime.date()
        == requested_date
    )


# ============================================================
# FIND EXACT SPORTS GAME
# ============================================================

def find_exact_games(
    home_team,
    away_team,
    requested_date=None
):

    queries = [

        f"{home_team} {away_team}",

        home_team,

        away_team,
    ]

    candidates = {}

    print(
        "\nSearching Polymarket..."
    )

    for query in queries:

        results = public_search(
            query
        )

        for event in results:

            event_id = str(
                event.get(
                    "id",
                    ""
                )
            )

            if event_id:

                candidates[
                    event_id
                ] = event

    print(
        f"Search candidates inspected: "
        f"{len(candidates)}"
    )

    exact_games = []

    for event_id in candidates:

        event = get_event(
            event_id
        )

        if not isinstance(
            event,
            dict
        ):
            continue

        # Must be a real structured game.
        if not is_exact_game_event(
            event,
            home_team,
            away_team
        ):
            continue

        # Optional date restriction.
        if not matches_requested_date(
            event,
            requested_date
        ):
            continue

        # Ignore closed events.
        if event.get(
            "closed",
            False
        ):
            continue

        exact_games.append(
            event
        )

    # Sort closest/upcoming event first.
    def sort_value(
        event
    ):

        dt = parse_event_datetime(
            event
        )

        if dt is None:

            return datetime.max.replace(
                tzinfo=timezone.utc
            )

        return dt

    exact_games.sort(
        key=sort_value
    )

    return exact_games


# ============================================================
# MARKET TYPE
# ============================================================

def sports_market_type(
    market
):

    value = market.get(
        "sportsMarketType"
    )

    if value:
        return normalize_text(
            value
        )

    sports = market.get(
        "sports"
    )

    if isinstance(
        sports,
        dict
    ):

        value = sports.get(
            "sportsMarketType"
        )

        if value:

            return normalize_text(
                value
            )

    return ""


# ============================================================
# IS MONEYLINE
# ============================================================

def is_moneyline_market(
    market
):

    market_type = (
        sports_market_type(
            market
        )
    )

    if "moneyline" in market_type:
        return True

    question = normalize_text(
        market.get(
            "question",
            ""
        )
    )

    # Fallback for Gamma objects where sportsMarketType
    # is absent.
    banned = [
        "exact score",
        "first half",
        "second half",
        "total goals",
        "over",
        "under",
        "both teams",
        "first team to score",
        "spread",
        "handicap",
    ]

    if any(
        phrase in question
        for phrase in banned
    ):
        return False

    return True


# ============================================================
# OPEN MONEYLINE MARKETS
# ============================================================

def get_moneyline_markets(
    event
):

    markets = (
        event.get(
            "markets"
        )
        or []
    )

    open_markets = []

    for market in markets:

        if not isinstance(
            market,
            dict
        ):
            continue

        if market.get(
            "closed",
            False
        ):
            continue

        if not market.get(
            "active",
            True
        ):
            continue

        if not is_moneyline_market(
            market
        ):
            continue

        open_markets.append(
            market
        )

    return open_markets


# ============================================================
# CLOB ORDER BOOK
# ============================================================

def get_order_book(
    token_id
):

    if not token_id:
        return None

    data = get_json(
        f"{CLOB_BASE_URL}/book",
        params={
            "token_id":
                token_id
        }
    )

    if not isinstance(
        data,
        dict
    ):
        return None

    return data


# ============================================================
# BEST BID / ASK
# ============================================================

def best_prices(
    order_book
):

    if not isinstance(
        order_book,
        dict
    ):
        return (
            None,
            None,
            None,
            None,
        )

    bids = (
        order_book.get(
            "bids"
        )
        or []
    )

    asks = (
        order_book.get(
            "asks"
        )
        or []
    )

    bid_levels = []
    ask_levels = []

    for level in bids:

        try:

            bid_levels.append(
                (
                    float(
                        level["price"]
                    ),
                    float(
                        level.get(
                            "size",
                            0
                        )
                    )
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            pass

    for level in asks:

        try:

            ask_levels.append(
                (
                    float(
                        level["price"]
                    ),
                    float(
                        level.get(
                            "size",
                            0
                        )
                    )
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            pass

    if bid_levels:

        best_bid = max(
            bid_levels,
            key=lambda x: x[0]
        )

    else:

        best_bid = (
            None,
            None
        )

    if ask_levels:

        best_ask = min(
            ask_levels,
            key=lambda x: x[0]
        )

    else:

        best_ask = (
            None,
            None
        )

    return (
        best_bid[0],
        best_bid[1],
        best_ask[0],
        best_ask[1],
    )


# ============================================================
# FORMAT
# ============================================================

def probability(
    value
):

    if value is None:
        return "N/A"

    try:

        return (
            f"{float(value):.1%}"
        )

    except (
        TypeError,
        ValueError
    ):

        return "N/A"


def amount(
    value
):

    if value is None:
        return "N/A"

    try:

        return (
            f"{float(value):,.1f}"
        )

    except (
        TypeError,
        ValueError
    ):

        return "N/A"


# ============================================================
# DISPLAY CLOB TOKEN
# ============================================================

def display_token(
    outcome,
    token_id
):

    book = get_order_book(
        token_id
    )

    if book is None:

        print(
            f"    {outcome:<12} "
            "Order book unavailable"
        )

        return

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

    print(
        f"\n    {outcome}"
    )

    print(
        f"      Best bid: "
        f"{probability(bid)}"
        f"   size "
        f"{amount(bid_size)}"
    )

    print(
        f"      Best ask: "
        f"{probability(ask)}"
        f"   size "
        f"{amount(ask_size)}"
    )

    print(
        f"      Spread:   "
        f"{probability(spread)}"
    )


# ============================================================
# DISPLAY MARKET
# ============================================================

def display_market(
    market,
    number
):

    question = (
        market.get(
            "question"
        )
        or
        "Unnamed market"
    )

    outcomes = parse_array(
        market.get(
            "outcomes"
        )
    )

    prices = parse_array(
        market.get(
            "outcomePrices"
        )
    )

    token_ids = parse_array(
        market.get(
            "clobTokenIds"
        )
    )

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"MARKET {number}"
    )

    print(
        "-" * 72
    )

    print(
        question
    )

    market_type = sports_market_type(
        market
    )

    if market_type:

        print(
            f"Type: "
            f"{market_type}"
        )

    print(
        "\nDisplayed prices:"
    )

    for index, outcome in enumerate(
        outcomes
    ):

        if index < len(
            prices
        ):

            displayed_price = (
                prices[index]
            )

        else:

            displayed_price = None

        print(
            f"    "
            f"{str(outcome):<15}"
            f"{probability(displayed_price)}"
        )

    print(
        "\nExecutable order book:"
    )

    for index, token_id in enumerate(
        token_ids
    ):

        if index < len(
            outcomes
        ):

            outcome = (
                outcomes[index]
            )

        else:

            outcome = (
                f"Outcome {index + 1}"
            )

        display_token(
            str(outcome),
            token_id
        )


# ============================================================
# DISPLAY GAME
# ============================================================

def display_game(
    event
):

    title = (
        event.get(
            "title"
        )
        or
        "Unnamed soccer game"
    )

    event_datetime = (
        parse_event_datetime(
            event
        )
    )

    game_id = get_game_id(
        event
    )

    teams = get_event_teams(
        event
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "EXACT SOCCER GAME FOUND"
    )

    print(
        "=" * 78
    )

    print(
        title
    )

    if event_datetime:

        print(
            "Kickoff: "
            f"{event_datetime.isoformat()}"
        )

    print(
        f"Game ID: "
        f"{game_id}"
    )

    print(
        "\nSTRUCTURED TEAMS"
    )

    print(
        "-" * 40
    )

    for team in teams:

        if not isinstance(
            team,
            dict
        ):
            continue

        print(
            f"{str(team.get('ordering', '?')).upper():<6}"
            f"{team.get('name', 'Unknown')}"
        )

    markets = get_moneyline_markets(
        event
    )

    print(
        f"\nOpen match-result markets: "
        f"{len(markets)}"
    )

    if not markets:

        print(
            "\nThe game exists on Polymarket, "
            "but no open moneyline markets "
            "were found."
        )

        return

    for number, market in enumerate(
        markets,
        start=1
    ):

        display_market(
            market,
            number
        )


# ============================================================
# DATE INPUT
# ============================================================

def ask_date():

    value = input(
        "Match date YYYY-MM-DD "
        "(Enter to skip): "
    ).strip()

    if value == "":
        return None

    try:

        return datetime.strptime(
            value,
            "%Y-%m-%d"
        ).date()

    except ValueError:

        print(
            "Invalid date. "
            "Ignoring date filter."
        )

        return None


# ============================================================
# MAIN
# ============================================================

def run():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PHASE 3A — STRICT POLYMARKET SOCCER SCANNER"
    )

    print(
        "=" * 78
    )

    print(
        "\nREAD-ONLY MODE"
    )

    print(
        "Only structured soccer game events "
        "will be accepted."
    )

    print(
        "Season futures and unrelated "
        "prediction markets are ignored."
    )

    while True:

        print(
            "\n"
            + "-" * 78
        )

        home_team = input(
            "Home team: "
        ).strip()

        away_team = input(
            "Away team: "
        ).strip()

        if (
            not home_team
            or not away_team
        ):

            print(
                "Enter both teams."
            )

            continue

        requested_date = (
            ask_date()
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"SEARCHING FOR EXACT GAME:"
        )

        print(
            f"{home_team} vs {away_team}"
        )

        if requested_date:

            print(
                f"Date: "
                f"{requested_date}"
            )

        print(
            "=" * 78
        )

        games = find_exact_games(
            home_team,
            away_team,
            requested_date
        )

        if not games:

            print(
                "\nNO EXACT STRUCTURED "
                "POLYMARKET GAME FOUND."
            )

            print(
                "\nThis means the scanner "
                "did not find a Polymarket "
                "sports event whose structured "
                "home and away teams match "
                "the teams you entered."
            )

            print(
                "\nIt will NOT show season "
                "winner markets or other "
                "future predictions instead."
            )

        else:

            print(
                f"\nExact games found: "
                f"{len(games)}"
            )

            # If no date was entered and multiple
            # fixtures somehow exist, show only
            # the earliest matching event.
            display_game(
                games[0]
            )

        again = input(
            "\nSearch another game? "
            "(y/n): "
        ).strip().lower()

        if again not in {
            "y",
            "yes"
        }:

            break

    print(
        "\nScanner closed."
    )


if __name__ == "__main__":
    run()
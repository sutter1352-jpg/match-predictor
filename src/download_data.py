from pathlib import Path
import requests

from .config import BASE_URL, LEAGUES, RAW_DIR, SEASONS


def download_file(url: str, destination: Path) -> None:
    """Download one CSV unless it already exists."""
    if destination.exists():
        print(f"SKIP  {destination.name}")
        return

    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "soccer-prediction-research/1.0"},
    )
    response.raise_for_status()

    destination.write_bytes(response.content)
    print(f"DOWN  {destination.name}")


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        for league_code in LEAGUES:
            url = BASE_URL.format(season=season, league=league_code)
            destination = RAW_DIR / f"{season}_{league_code}.csv"

            try:
                download_file(url, destination)
            except requests.RequestException as exc:
                print(f"FAIL  {season} {league_code}: {exc}")


if __name__ == "__main__":
    download_all()

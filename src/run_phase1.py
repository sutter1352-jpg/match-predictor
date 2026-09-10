from src.download_data import download_all
from src.build_dataset import build_dataset
from src.elo import run as run_elo
from src.validate_data import validate


def main():
    print("\n1/4 Downloading historical data...")
    download_all()

    print("\n2/4 Building clean dataset...")
    build_dataset()

    print("\n3/4 Calculating pre-match Elo ratings...")
    run_elo()

    print("\n4/4 Validating dataset...")
    validate()

    print("\nPHASE 1 FOUNDATION COMPLETE")


if __name__ == "__main__":
    main()

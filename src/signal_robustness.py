import numpy as np
import pandas as pd

from .config import PROCESSED_DIR


RANDOM_SEED = 42
BOOTSTRAPS = 10000

THRESHOLDS = [
    0.0,
    0.0025,
    0.0050,
]


def bootstrap_ci(values, rng):

    values = np.asarray(values)

    samples = []

    n = len(values)

    for _ in range(BOOTSTRAPS):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        samples.append(
            np.mean(sample)
        )

    samples = np.asarray(samples)

    lower = np.percentile(
        samples,
        2.5
    )

    upper = np.percentile(
        samples,
        97.5
    )

    return lower, upper


def run():

    path = (
        PROCESSED_DIR
        / "ridge_signals.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print(
        "\nSIGNAL ROBUSTNESS"
    )

    print(
        "=" * 72
    )

    for threshold in THRESHOLDS:

        subset = df[
            df["predicted_move_abs"]
            >= threshold
        ].copy()

        direction = (
            subset["correct_direction"]
            .astype(float)
            .to_numpy()
        )

        clv = (
            subset["directional_clv"]
            .to_numpy()
        )

        direction_mean = (
            direction.mean()
        )

        clv_mean = (
            clv.mean()
        )

        (
            direction_low,
            direction_high
        ) = bootstrap_ci(
            direction,
            rng
        )

        (
            clv_low,
            clv_high
        ) = bootstrap_ci(
            clv,
            rng
        )

        # Probability from bootstrap samples
        # that mean CLV is <= zero

        bootstrap_clv = []

        for _ in range(BOOTSTRAPS):

            sample = rng.choice(
                clv,
                size=len(clv),
                replace=True
            )

            bootstrap_clv.append(
                sample.mean()
            )

        bootstrap_clv = np.array(
            bootstrap_clv
        )

        probability_nonpositive = (
            bootstrap_clv <= 0
        ).mean()

        print(
            f"\nThreshold >= {threshold:.2%}"
        )

        print(
            "-" * 40
        )

        print(
            f"Signals: {len(subset)}"
        )

        print(
            f"Direction accuracy: "
            f"{direction_mean:.2%}"
        )

        print(
            f"95% CI: "
            f"{direction_low:.2%} "
            f"to "
            f"{direction_high:.2%}"
        )

        print(
            f"Average CLV: "
            f"{clv_mean:.3%}"
        )

        print(
            f"95% CLV CI: "
            f"{clv_low:.3%} "
            f"to "
            f"{clv_high:.3%}"
        )

        print(
            f"Bootstrap probability "
            f"mean CLV <= 0: "
            f"{probability_nonpositive:.3f}"
        )


if __name__ == "__main__":
    run()
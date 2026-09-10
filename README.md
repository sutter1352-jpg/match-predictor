# Soccer Prediction Model — Phase 1

Phase 1 builds the historical-data foundation for:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

Training window: 2015/16 through 2025/26.

## What this version does

1. Downloads historical CSV match data.
2. Combines the five leagues into one clean dataset.
3. Creates Home / Draw / Away target labels.
4. Calculates pre-match Elo ratings without future leakage.
5. Runs basic validation checks.

Champions League and Europa League are intentionally added later because
cross-league ratings and competition data need special treatment.

## Setup

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Run Phase 1:

```bash
python run_phase1.py
```

## Output

The important files will be:

```text
data/processed/matches_clean.csv
data/processed/matches_with_elo.csv
```

`matches_with_elo.csv` will become the input to the first prediction models.

## Important modeling rule

The Elo values stored on each match are the ratings immediately BEFORE that
match was played.

That is essential because using post-match information would leak the result
into the training data and make backtest performance look artificially good.

## Next step

Build rolling pre-match features:

- last 5 matches points
- goals scored / conceded
- shots / shots on target
- home and away form
- rest days

Then build a time-aware baseline classifier and Dixon-Coles/Poisson model.

# College Football Bowl Predictor (UE24CS352A ML Mini-Project, Problem 88, Team G-29)

Predicts the winner of every college football bowl game **and a confidence** (predicted point margin → win
probability), inspired by Stanford CS229 (2015) project #101. We re-build the data pipeline from public
play-by-play data, engineer regular-season team features, and compare 8 models under leave-one-season-out CV.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (≈3 min total; needs internet for the first step)
```bash
python src/build_dataset.py 2005 2020   # downloads play-by-play from github.com/sportsdataverse/cfbfastR-data, builds data/processed/*.csv
cd src && python train.py && cd ..       # LOSO-CV of all models, figures + tables in outputs/, final model in models/
python src/predict.py slate 2019         # pick'em sheet for a season (out-of-fold)
python src/predict.py matchup 2018 "Alabama" "Oklahoma"
streamlit run src/app.py                 # interactive demo
```
Pre-built `data/processed/`, `models/` and `outputs/` are committed so the demo works without re-downloading.

## Layout
| Path | Purpose |
|---|---|
| `src/build_dataset.py` | play-by-play → per-game scores and per-team-game box-score stats |
| `src/features.py` | season aggregates (offense, defense, turnovers, special teams, Elo); **regular season only → no leakage** |
| `src/models.py` | OLS / Ridge / Lasso / Logistic / Random forest / Gradient boosting / Ensemble on *difference* features |
| `src/train.py` | leave-one-season-out CV, pick'em metric, significance test, figures, final model |
| `src/predictor.py`, `predict.py`, `app.py` | inference library, CLI, Streamlit demo |

## Method in one paragraph
Each bowl game is represented by `features(team1) − features(team2)` (36 features). Linear models have no intercept, so
swapping the teams exactly flips the prediction. Regularisation strength is tuned *inside* each training fold.
Evaluation = leave-one-season-out over 16 seasons (566 bowl games, 552 with features).

## Results (552 games)
| Model | Winner acc. | Pick'em % of max |
|---|---|---|
| Coin flip | 50.0% | 50.0% |
| Higher win % | 53.2% | 53.7% |
| Higher Elo | 58.2% | 61.7% |
| **Ridge (deployed)** | **62.9%** | 64.6% |
| Ridge, scoring+Elo features only | 63.1% | 67.6% |
| Random forest / Ensemble | 61.1% | 63.8% / 64.1% |

Ridge beats the Elo baseline (exact sign test p = 0.029, one of five comparisons, so treat as suggestive).
Honest caveats are in the write-up.

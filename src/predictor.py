"""Inference helpers shared by the CLI (predict.py) and the Streamlit demo (app.py)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

# Ensure backward compatibility when unpickling models across scikit-learn versions
try:
    import sklearn._loss._loss as _loss
    sys.modules['_loss'] = _loss
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import models
    sys.modules['models'] = models
except Exception:
    pass

import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(HERE, "..")


class BowlPredictor:
    def __init__(self):
        art = joblib.load(os.path.join(ROOT, "models", "final_model.joblib"))
        self.cols, self.ridge = art["cols"], art["models"]["Ridge"]
        self.tab = pd.read_csv(os.path.join(ROOT, "data", "processed", "team_season_features.csv"))
        self.names = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "models", "team_names.json")))["names"].items()}
        self.games = pd.read_csv(os.path.join(ROOT, "data", "processed", "games.csv"))
        self.oof = pd.read_csv(os.path.join(ROOT, "outputs", "oof_predictions.csv"))
        # map predicted margin -> win probability with a single calibration slope fitted on out-of-fold predictions
        y = (self.oof["margin"] > 0).astype(int)
        self.slope = float(LogisticRegression(fit_intercept=False, C=1e6).fit(self.oof[["Ridge"]], y).coef_[0, 0])

    # ---------- teams ----------
    def teams_in(self, season):
        ids = self.tab.loc[self.tab.season == season, "team"]
        return sorted((self.names.get(int(i), str(i)), int(i)) for i in ids)

    def find_team(self, query, season):
        cand = self.teams_in(season)
        q = query.strip().lower()
        for test in (lambda n: n.lower() == q, lambda n: n.lower().startswith(q), lambda n: q in n.lower()):
            hit = [(n, i) for n, i in cand if test(n)]
            if len(hit) == 1: return hit[0]
            if len(hit) > 1:
                exact = [h for h in hit if h[0].lower() == q]
                if exact: return exact[0]
                raise ValueError(f"'{query}' is ambiguous: {[h[0] for h in hit][:8]}")
        raise ValueError(f"No team matching '{query}' with a {season} regular season in the dataset")

    def _vec(self, season, team):
        r = self.tab[(self.tab.season == season) & (self.tab.team == team)]
        if r.empty: raise ValueError("team has no features for that season")
        return r.iloc[0][self.cols].to_numpy(float)

    # ---------- predictions ----------
    def matchup(self, season, name1, name2):
        (n1, i1), (n2, i2) = self.find_team(name1, season), self.find_team(name2, season)
        x = (self._vec(season, i1) - self._vec(season, i2))[None, :]
        margin = float(self.ridge.score(x)[0])
        p1 = 1 / (1 + np.exp(-self.slope * margin))
        z = self.ridge.sc.transform(x)[0] * self.ridge.coef_
        drivers = pd.Series(z, index=self.cols).sort_values(key=np.abs, ascending=False).head(6)
        f1, f2 = self.tab[(self.tab.season == season) & (self.tab.team == i1)].iloc[0], \
                 self.tab[(self.tab.season == season) & (self.tab.team == i2)].iloc[0]
        return dict(team1=n1, team2=n2, season=season, margin=margin, p_team1=p1,
                    winner=n1 if margin > 0 else n2, win_prob=max(p1, 1 - p1),
                    drivers=drivers, f1=f1, f2=f2)

    def slate(self, season):
        """Out-of-fold (leave-that-season-out) picks for every bowl game of a season, with pick'em confidence ranks."""
        o = self.oof[self.oof.season == season].copy()
        if o.empty: raise ValueError("no bowl games for that season")
        o["pick"] = np.where(o["Ridge"] > 0, o.home_team, o.away_team)
        o["pred_margin"] = o["Ridge"].abs()
        o["win_prob"] = 1 / (1 + np.exp(-self.slope * o["pred_margin"]))
        o = o.sort_values("pred_margin").reset_index(drop=True)
        o["confidence_pts"] = np.arange(1, len(o) + 1)
        o["actual_winner"] = np.where(o.margin > 0, o.home_team, o.away_team)
        o["correct"] = o["pick"] == o["actual_winner"]
        o["points"] = np.where(o.correct, o.confidence_pts, 0)
        o["game"] = o.home_team + " vs " + o.away_team
        o["final"] = o.home_points.astype(int).astype(str) + "-" + o.away_points.astype(int).astype(str)
        return o[["confidence_pts", "game", "pick", "pred_margin", "win_prob", "actual_winner", "final", "correct", "points"]]

"""Model zoo for the bowl predictor.

All models work on DIFFERENCE features  x = features(team1) - features(team2)  and output a signed score
(positive => team1 wins). Linear models have no intercept, so they are exactly antisymmetric: swapping the two
teams flips the prediction. Tree models are trained on both orientations and symmetrised at prediction time.
Hyper-parameters are tuned with leave-one-season-out CV *inside* the training data (no leakage).
"""
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def season_splits(seasons):
    seasons = np.asarray(seasons)
    return [(np.where(seasons != s)[0], np.where(seasons == s)[0]) for s in np.unique(seasons)]


class LinearModel:
    def __init__(self, kind="ridge", cols=None, seed=0):
        self.kind, self.cols, self.seed = kind, cols, seed

    def _sel(self, X):
        return X if self.cols is None else X[:, self.cols]

    def fit(self, X, y, seasons):
        X = self._sel(X)
        rng = np.random.default_rng(self.seed)
        s = rng.choice([-1.0, 1.0], len(y))                      # random orientation (team1/team2 is arbitrary)
        Xs, ys = X * s[:, None], y * s
        self.sc = StandardScaler(with_mean=False).fit(Xs)
        Z = self.sc.transform(Xs)
        cv = season_splits(seasons)
        if self.kind == "ols":
            self.m = LinearRegression(fit_intercept=False).fit(Z, ys)
        elif self.kind == "ridge":
            self.m = RidgeCV(alphas=np.logspace(0, 4, 25), fit_intercept=False, cv=cv).fit(Z, ys)
        elif self.kind == "lasso":
            self.m = LassoCV(alphas=np.logspace(-1.5, 1, 30), fit_intercept=False, cv=cv, max_iter=50000).fit(Z, ys)
        elif self.kind == "logit":
            self.m = LogisticRegressionCV(Cs=np.logspace(-4, 0, 12), fit_intercept=False, cv=cv,
                                          scoring="neg_log_loss", max_iter=5000).fit(Z, (ys > 0).astype(int))
        return self

    def score(self, X):
        Z = self.sc.transform(self._sel(X))
        return self.m.decision_function(Z) if self.kind == "logit" else self.m.predict(Z)

    @property
    def coef_(self):
        return self.m.coef_.ravel()


class RuleModel:
    """Un-fitted rule: pick the team with the larger value of one feature (e.g. higher Elo)."""
    def __init__(self, col): self.col = col
    def fit(self, X, y, seasons=None): return self
    def score(self, X): return X[:, self.col]


class TreeModel:
    def __init__(self, kind="rf", seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, X, y, seasons=None):
        Xa, ya = np.vstack([X, -X]), np.concatenate([y, -y])
        if self.kind == "rf":
            self.m = RandomForestRegressor(n_estimators=400, min_samples_leaf=12, max_features=0.3,
                                           n_jobs=-1, random_state=self.seed)
        else:
            self.m = GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=2,
                                               subsample=0.7, random_state=self.seed)
        self.m.fit(Xa, ya)
        return self

    def score(self, X):
        return (self.m.predict(X) - self.m.predict(-X)) / 2


class Ensemble:
    """Average of the margin-scale models."""
    def __init__(self, members):
        self.members = members

    def fit(self, X, y, seasons):
        for m in self.members: m.fit(X, y, seasons)
        return self

    def score(self, X):
        return np.mean([m.score(X) for m in self.members], axis=0)


def make_models(feature_names):
    ix = {c: i for i, c in enumerate(feature_names)}
    simple = [ix[c] for c in ("elo", "win_pct", "ppg", "papg", "margin_pg")]
    eff = [ix[c] for c in feature_names if c not in ("elo", "win_pct", "ppg", "papg", "margin_pg", "margin_std")]
    return {
        "Baseline: higher Elo":        lambda: RuleModel(ix["elo"]),
        "Baseline: higher win %":      lambda: RuleModel(ix["win_pct"]),
        "Linear regression (all)":     lambda: LinearModel("ols"),
        "Ridge":                       lambda: LinearModel("ridge"),
        "Lasso":                       lambda: LinearModel("lasso"),
        "Logistic regression":         lambda: LinearModel("logit"),
        "Random forest":               lambda: TreeModel("rf"),
        "Gradient boosting":           lambda: TreeModel("gb"),
        "Ensemble (Ridge+RF+GB)":      lambda: Ensemble([LinearModel("ridge"), TreeModel("rf"), TreeModel("gb")]),
        "Ridge: scoring/Elo only":     lambda: LinearModel("ridge", cols=simple),
        "Ridge: efficiency stats only": lambda: LinearModel("ridge", cols=eff),
    }

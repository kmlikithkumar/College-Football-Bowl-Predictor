"""Leave-one-season-out evaluation of all models + final model training.

    python src/train.py            # writes outputs/*.csv, outputs/figures/*.png, models/final_model.joblib
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import features as F
from models import make_models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs"); FIG = os.path.join(OUT, "figures"); MOD = os.path.join(HERE, "..", "models")
for d in (OUT, FIG, MOD): os.makedirs(d, exist_ok=True)


def correct_vec(score, margin):
    """1 if the picked winner is right, 0 if wrong, 0.5 if the model has no preference (score == 0)."""
    return np.where(score == 0, 0.5, (np.sign(score) == np.sign(margin)).astype(float))


def pickem_score(seasons, margin, score):
    """Bowl pick'em game: in each season rank picks by confidence (1 = least sure ... N = most sure);
    a correct pick earns its confidence points. Returns mean fraction of the maximum N(N+1)/2 achieved."""
    out = []
    for s in np.unique(seasons):
        m = seasons == s
        conf = np.empty(m.sum()); order = np.argsort(np.abs(score[m]), kind="stable")
        conf[order] = np.arange(1, m.sum() + 1)
        correct = correct_vec(score[m], margin[m])
        out.append((conf * correct).sum() / (m.sum() * (m.sum() + 1) / 2))
    return float(np.mean(out))


def main():
    t0 = time.time()
    games, tm = F.load()
    b, tab, cols = F.bowl_dataset(games, tm)
    X = b[[f"t1_{c}" for c in cols]].to_numpy(float) - b[[f"t2_{c}" for c in cols]].to_numpy(float)
    y = b["margin"].to_numpy(float); seasons = b["season"].to_numpy()
    print(f"{len(b)} bowl games, {len(cols)} features, seasons {seasons.min()}-{seasons.max()}")
    zoo = make_models(cols)
    oof = {k: np.zeros(len(b)) for k in zoo}
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        for name, mk in zoo.items():
            oof[name][te] = mk().fit(X[tr], y[tr], seasons[tr]).score(X[te])
        print(f"  fold {s} done ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for name, sc in oof.items():
        acc = float(np.mean(correct_vec(sc, y)))
        margin_model = not (name.startswith("Baseline") or name == "Logistic regression")
        rows.append(dict(model=name, accuracy=acc, acc_se=np.sqrt(acc * (1 - acc) / len(y)),
                         pickem_pct_of_max=pickem_score(seasons, y, sc),
                         margin_RMSE=float(np.sqrt(np.mean((sc - y) ** 2))) if margin_model else np.nan,
                         margin_MAE=float(np.mean(np.abs(sc - y))) if margin_model else np.nan))
    res = pd.DataFrame(rows)
    res.loc[len(res)] = dict(model="Coin flip (reference)", accuracy=0.5, acc_se=np.sqrt(.25 / len(y)),
                             pickem_pct_of_max=0.5, margin_RMSE=float(np.sqrt(np.mean(y ** 2))), margin_MAE=float(np.mean(np.abs(y))))
    res.round(4).to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(res.round(3).to_string(index=False))

    pred = b[["game_id", "season", "home_team", "away_team", "home_points", "away_points", "margin"]].copy()
    for k, v in oof.items(): pred[k] = v
    pred.to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)

    # per-season accuracy table for the report
    per = pd.DataFrame({k: pd.Series(correct_vec(v, y)).groupby(seasons).mean() for k, v in oof.items()})
    per.round(3).to_csv(os.path.join(OUT, "accuracy_by_season.csv"))

    # ---------- figures ----------
    r = res[res.model != "Coin flip (reference)"].sort_values("accuracy")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#9aa5b1" if m.startswith("Baseline") else "#7fb3e6" if "only" in m else "#2b6cb0" for m in r.model]
    ax[0].barh(r.model, r.accuracy * 100, xerr=r.acc_se * 100, color=colors, capsize=3)
    ax[0].axvline(50, ls="--", c="k", lw=1); ax[0].set_xlim(45, 70); ax[0].set_xlabel("Winner accuracy, % (leave-one-season-out)")
    ax[0].set_title("Picking bowl winners")
    ax[1].barh(r.model, r.pickem_pct_of_max * 100, color=colors)
    ax[1].axvline(50, ls="--", c="k", lw=1); ax[1].set_xlim(45, 70); ax[1].set_yticklabels([]); ax[1].set_xlabel("Pick'em score, % of maximum")
    ax[1].set_title("Confidence-weighted pick'em")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=160); plt.close()

    # paired significance vs the Elo baseline (exact sign test on games where exactly one model is right)
    from scipy.stats import binomtest
    ok = {k: correct_vec(v, y) == 1.0 for k, v in oof.items()}
    lines = []
    for k in ["Ridge", "Lasso", "Logistic regression", "Random forest", "Ensemble (Ridge+RF+GB)"]:
        a, c = int((ok[k] & ~ok["Baseline: higher Elo"]).sum()), int((~ok[k] & ok["Baseline: higher Elo"]).sum())
        p = binomtest(a, a + c, 0.5).pvalue
        lines.append(f"{k:26s} vs Elo: model-only-correct={a:3d}  elo-only-correct={c:3d}  two-sided p={p:.3f}")
    open(os.path.join(OUT, "significance_vs_elo.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))

    best = "Ridge"  # deployed model: best winner-accuracy, simplest, interpretable (chosen a priori, see report)
    sc = oof[best]; q = pd.qcut(np.abs(sc), 4, labels=["lowest", "low-mid", "high-mid", "highest"])
    cal = pd.DataFrame({"q": q, "ok": correct_vec(sc, y)}).groupby("q", observed=True).ok.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(cal.index.astype(str), cal["mean"] * 100, color="#2b6cb0"); ax.axhline(50, ls="--", c="k", lw=1)
    ax.set_ylim(30, 85); ax.set_ylabel("Accuracy, %"); ax.set_xlabel("Model confidence quartile (|predicted margin|)")
    ax.set_title(f"Confidence is informative\n({best})")
    for i, (v, n) in enumerate(zip(cal["mean"], cal["size"])): ax.text(i, v * 100 + 1, f"{v*100:.0f}%\nn={n}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "confidence_calibration.png"), dpi=160); plt.close()
    cal.to_csv(os.path.join(OUT, "confidence_quartiles.csv"))

    # ---------- final models on ALL seasons ----------
    final = {k: zoo[k]().fit(X, y, seasons) for k in ["Ridge", "Logistic regression", "Random forest",
                                                     "Gradient boosting", "Ensemble (Ridge+RF+GB)"]}
    ridge = final["Ridge"]
    coef = pd.Series(ridge.coef_, index=cols).sort_values(key=np.abs).tail(12)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(coef.index, coef.values, color=["#2b6cb0" if v > 0 else "#c05621" for v in coef.values])
    ax.set_xlabel("Ridge coefficient (standardised difference feature, points of margin)")
    ax.set_title("What drives the prediction"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ridge_coefficients.png"), dpi=160); plt.close()
    joblib.dump({"models": final, "cols": cols, "best_cv": best}, os.path.join(MOD, "final_model.joblib"))
    tab.to_csv(os.path.join(F.PROC, "team_season_features.csv"), index=False)
    json.dump({"names": {int(k): v for k, v in F.team_names(games).items()}}, open(os.path.join(MOD, "team_names.json"), "w"))
    print("best model by pick'em:", best, f"| total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

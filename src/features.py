"""Feature engineering for the College Football Bowl Predictor.

Everything here uses REGULAR-SEASON information only (season_type == 2), so there is no leakage
from the bowl game itself. Output: one row per (season, team) with season-long averages.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")

K_ELO, HFA, REVERT = 20.0, 55.0, 1 / 3


def load():
    games = pd.read_csv(os.path.join(PROC, "games.csv"))
    tm = pd.read_csv(os.path.join(PROC, "team_game_stats.csv")).dropna(subset=["team"])
    tm["team"] = tm["team"].astype(int)
    return games, tm


def team_names(games):
    a = games[["home_id", "home_team"]].rename(columns={"home_id": "team", "home_team": "name"})
    b = games[["away_id", "away_team"]].rename(columns={"away_id": "team", "away_team": "name"})
    n = pd.concat([a, b]).drop_duplicates("team", keep="last")
    return dict(zip(n.team, n.name))


def compute_elo(games):
    """End-of-regular-season Elo per (season, team), carried across seasons with regression to the mean."""
    reg = games[games.season_type == 2].sort_values(["season", "week", "game_id"])
    elo, rows, cur = {}, [], None
    for r in reg.itertuples():
        if r.season != cur:                      # new season: shrink everyone towards 1500
            if cur is not None:
                rows += [(cur, t, e) for t, e in elo.items()]
            elo = {t: 1500 + (1 - REVERT) * (e - 1500) for t, e in elo.items()}
            cur = r.season
        eh, ea = elo.get(r.home_id, 1500.0), elo.get(r.away_id, 1500.0)
        p_home = 1 / (1 + 10 ** (-((eh + HFA) - ea) / 400))
        s_home = 1.0 if r.home_points > r.away_points else 0.0 if r.home_points < r.away_points else 0.5
        mov = np.log(abs(r.home_points - r.away_points) + 1) + 1          # margin-of-victory multiplier
        d = K_ELO * mov * (s_home - p_home)
        elo[r.home_id], elo[r.away_id] = eh + d, ea - d
    rows += [(cur, t, e) for t, e in elo.items()]
    return pd.DataFrame(rows, columns=["season", "team", "elo"])


def team_season_table(games, tm):
    reg = games[games.season_type == 2]
    h = reg.rename(columns={"home_id": "team", "away_id": "opp", "home_points": "pts", "away_points": "opp_pts"})
    a = reg.rename(columns={"away_id": "team", "home_id": "opp", "away_points": "pts", "home_points": "opp_pts"})
    keep = ["game_id", "season", "team", "opp", "pts", "opp_pts"]
    res = pd.concat([h[keep], a[keep]])
    res["win"] = (res.pts > res.opp_pts).astype(float)
    res["margin"] = res.pts - res.opp_pts
    d = res.merge(tm.drop(columns="season"), on=["game_id", "team"], how="left")
    g = d.groupby(["season", "team"])
    s = g.agg(games=("game_id", "size"), win_pct=("win", "mean"), ppg=("pts", "mean"), papg=("opp_pts", "mean"),
              margin_std=("margin", "std"),
              rush_att=("off_rush_att", "sum"), rush_yds=("off_rush_yds", "sum"),
              pass_att=("off_pass_att", "sum"), pass_comp=("off_pass_comp", "sum"), pass_yds=("off_pass_yds", "sum"),
              sacks=("off_sacks", "sum"), ints=("off_ints", "sum"), fumbles=("off_fumbles", "mean"),
              off_epa=("off_epa_play", "mean"), off_success=("off_success", "mean"), off_havoc=("off_havoc", "mean"),
              d_rush_att=("def_rush_att", "sum"), d_rush_yds=("def_rush_yds", "sum"),
              d_pass_att=("def_pass_att", "sum"), d_pass_comp=("def_pass_comp", "sum"), d_pass_yds=("def_pass_yds", "sum"),
              d_sacks=("def_sacks", "sum"), d_ints=("def_ints", "sum"), d_fumbles=("def_fumbles", "mean"),
              def_epa=("def_epa_play", "mean"), def_success=("def_success", "mean"), def_havoc=("def_havoc", "mean"),
              pen_cnt=("pen_cnt", "mean"), punt_avg=("punt_avg", "mean"),
              fga=("fga", "sum"), fgm=("fgm", "sum")).reset_index()
    f = pd.DataFrame({"season": s.season, "team": s.team, "win_pct": s.win_pct, "ppg": s.ppg, "papg": s.papg,
                      "margin_pg": s.ppg - s.papg, "margin_std": s.margin_std,
                      "off_ypc": s.rush_yds / s.rush_att.clip(lower=1),
                      "off_ypa": s.pass_yds / s.pass_att.clip(lower=1),
                      "off_comp_pct": s.pass_comp / s.pass_att.clip(lower=1),
                      "off_sack_rate": s.sacks / (s.pass_att + s.sacks).clip(lower=1),
                      "off_int_rate": s.ints / s.pass_att.clip(lower=1),
                      "off_fumbles_pg": s.fumbles, "off_epa": s.off_epa, "off_success": s.off_success,
                      "off_havoc_allowed": s.off_havoc,
                      "def_ypc": s.d_rush_yds / s.d_rush_att.clip(lower=1),
                      "def_ypa": s.d_pass_yds / s.d_pass_att.clip(lower=1),
                      "def_comp_pct": s.d_pass_comp / s.d_pass_att.clip(lower=1),
                      "def_sack_rate": s.d_sacks / (s.d_pass_att + s.d_sacks).clip(lower=1),
                      "def_int_rate": s.d_ints / s.d_pass_att.clip(lower=1),
                      "def_fumbles_pg": s.d_fumbles, "def_epa": s.def_epa, "def_success": s.def_success,
                      "def_havoc": s.def_havoc,
                      "pen_pg": s.pen_cnt, "punt_avg": s.punt_avg,
                      "fg_pct": s.fgm / s.fga.where(s.fga >= 3)})
    f["turnover_margin_pg"] = (s.d_ints / s.games + s.d_fumbles) - (s.ints / s.games + s.fumbles)
    f = f.merge(compute_elo(games), on=["season", "team"], how="left")
    f["games"] = s.games.values
    f = f[f.games >= 8].copy()                          # only teams with a real regular season
    num = [c for c in f.columns if c not in ("season", "team", "games")]
    f[num] = f.groupby("season")[num].transform(lambda x: x.fillna(x.median()))   # rare missing -> season median
    return f.reset_index(drop=True)


FEATURES = None


def bowl_dataset(games=None, tm=None):
    """One row per bowl game: team1 (home id) vs team2, features of each, final margin."""
    if games is None:
        games, tm = load()
    tab = team_season_table(games, tm)
    cols = [c for c in tab.columns if c not in ("season", "team", "games")]
    b = games[games.season_type == 3].copy()
    b = b.merge(tab.add_prefix("t1_").rename(columns={"t1_season": "season", "t1_team": "home_id"}), on=["season", "home_id"])
    b = b.merge(tab.add_prefix("t2_").rename(columns={"t2_season": "season", "t2_team": "away_id"}), on=["season", "away_id"])
    b["margin"] = b.home_points - b.away_points
    b = b[b.margin != 0].reset_index(drop=True)
    return b, tab, cols


if __name__ == "__main__":
    b, tab, cols = bowl_dataset()
    print("bowl games with features:", len(b), "| team-seasons:", len(tab), "| features:", len(cols))
    print(tab.describe().T.head(12))

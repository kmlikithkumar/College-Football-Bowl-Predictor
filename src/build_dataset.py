"""Build the project dataset from sportsdataverse/cfbfastR-data play-by-play (public GitHub).

For every season we aggregate play-by-play into
  * data/processed/games.csv           one row per game (regular season + bowls) with final score
  * data/processed/team_game_stats.csv one row per team per game (offense/defense/special teams box-score stats)

Usage:  python src/build_dataset.py 2005 2020
"""
import os, sys, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/master"
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "processed")

USE = ["game_id", "season", "season_type", "week", "posteam", "defteam", "home_team_id", "away_team_id",
       "home_team_name", "away_team_name", "home_score", "away_score", "rush", "pass", "pass_attempt",
       "complete_pass", "sack", "int", "fumble", "touchdown", "yds_rushed", "yds_receiving", "yds_sacked",
       "penalty_flag", "yds_penalty", "punt_play", "yds_punted",
       "field_goal_attempt", "fg_made", "epa", "epa_success", "havoc", "rz_play"]
BOOL = ["rush", "pass", "pass_attempt", "complete_pass", "sack", "int", "fumble", "touchdown", "penalty_flag",
        "punt_play", "field_goal_attempt", "fg_made", "havoc", "rz_play"]
NUM = ["yds_rushed", "yds_receiving", "yds_sacked", "yds_penalty", "yds_punted", "epa",
       "epa_success"]


def download(year):
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, f"pbp_{year}.csv.gz")
    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
        urllib.request.urlretrieve(f"{BASE}/pbp/csv/play_by_play_{year}.csv.gz", dest)
    return dest


def side(df, key, prefix):
    g = df.groupby(["game_id", key])
    o = pd.DataFrame({
        "plays": g.size(), "rush_att": g["rush"].sum(), "rush_yds": g["yds_rushed"].sum(),
        "dropbacks": g["pass"].sum(), "sacks": g["sack"].sum(),
        "pass_comp": g["complete_pass"].sum(), "pass_yds": g["yds_receiving"].sum(),
        "ints": g["int"].sum(), "fumbles": g["fumble"].sum(),
        "epa_play": g["epa"].mean(), "success": g["epa_success"].mean(), "havoc": g["havoc"].mean(),
    })
    o[f"pass_att"] = o["dropbacks"] - o["sacks"]          # sacks are counted as pass plays in the source data
    o.columns = [f"{prefix}_{c}" for c in o.columns]
    o.index.names = ["game_id", "team"]
    return o


def process(year):
    d = pd.read_csv(download(year), usecols=lambda c: c in USE, low_memory=False, encoding_errors="replace")
    for c in BOOL: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["yds_rushed", "yds_receiving", "yds_sacked", "yds_penalty"]:
        d[c] = d[c].fillna(0)
    # ---- games (final score = max running score) ----
    games = d.groupby("game_id").agg(
        season=("season", "first"), season_type=("season_type", "first"), week=("week", "first"),
        home_id=("home_team_id", "first"), away_id=("away_team_id", "first"),
        home_team=("home_team_name", "first"), away_team=("away_team_name", "first"),
        home_points=("home_score", "max"), away_points=("away_score", "max")).reset_index()
    # ---- team-game stats ----
    scrim = d[(d["rush"] == 1) | (d["pass"] == 1)]
    off, dfn = side(scrim, "posteam", "off"), side(scrim, "defteam", "def")
    pen = d[d["penalty_flag"] == 1].groupby(["game_id", "posteam"]).size().rename("pen_cnt").to_frame()
    pen.index.names = ["game_id", "team"]
    pu = d[d["punt_play"] == 1].groupby(["game_id", "posteam"])["yds_punted"].mean().rename("punt_avg").to_frame()
    pu.index.names = ["game_id", "team"]
    fg = d[d["field_goal_attempt"] == 1].groupby(["game_id", "posteam"]).agg(fga=("field_goal_attempt", "sum"), fgm=("fg_made", "sum"))
    fg.index.names = ["game_id", "team"]
    tm = off.join(dfn, how="outer").join(pen, how="left").join(pu, how="left").join(fg, how="left").reset_index()
    tm["season"] = year
    tm = tm[(tm["off_plays"].between(35, 175)) & (tm["def_plays"].between(35, 175))]
    return games, tm


if __name__ == "__main__":
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    G, T = [], []
    for y in range(y0, y1 + 1):
        try:
            g, t = process(y); G.append(g); T.append(t)
            print(y, "ok", len(g), "games,", int((g.season_type == 3).sum()), "bowl/postseason", flush=True)
        except Exception as e:
            print(y, "FAILED", repr(e)[:200], flush=True)
    pd.concat(G).to_csv(os.path.join(OUT, "games.csv"), index=False)
    pd.concat(T).to_csv(os.path.join(OUT, "team_game_stats.csv"), index=False)
    print("done")

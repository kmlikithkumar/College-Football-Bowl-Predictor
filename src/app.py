"""Streamlit demo:  streamlit run src/app.py"""
import streamlit as st
import pandas as pd
from predictor import BowlPredictor

st.set_page_config(page_title="College Football Bowl Predictor", layout="wide")


@st.cache_resource
def load():
    return BowlPredictor()


P = load()
st.title("🏈 College Football Bowl Predictor")
st.caption("Ridge regression on regular-season efficiency + Elo differences, trained on bowl games 2005-2020 "
           "(leave-one-season-out CV accuracy ≈ 63%).")
tab1, tab2 = st.tabs(["Head-to-head matchup", "Season pick'em sheet"])

with tab1:
    seasons = sorted(P.tab.season.unique())
    season = st.selectbox("Season (regular-season form used)", seasons, index=len(seasons) - 1)
    names = [n for n, _ in P.teams_in(season)]
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Team 1", names, index=0)
    t2 = c2.selectbox("Team 2", names, index=min(1, len(names) - 1))
    if t1 == t2:
        st.warning("Pick two different teams.")
    else:
        r = P.matchup(season, t1, t2)
        a, b, c = st.columns(3)
        a.metric("Predicted winner", r["winner"])
        b.metric("Win probability", f"{r['win_prob']*100:.0f}%")
        c.metric("Predicted margin", f"{abs(r['margin']):.1f} pts")
        st.progress(float(r["p_team1"]), text=f"{t1} {r['p_team1']*100:.0f}%  |  {t2} {(1-r['p_team1'])*100:.0f}%")
        st.subheader("Why? Biggest drivers (points of margin toward " + t1 + ")")
        st.bar_chart(r["drivers"])
        cols = ["win_pct", "ppg", "papg", "elo", "off_epa", "def_epa", "turnover_margin_pg"]
        st.dataframe(pd.DataFrame({t1: r["f1"][cols], t2: r["f2"][cols]}).round(2))

with tab2:
    s = st.selectbox("Bowl season", sorted(P.oof.season.unique()), index=len(P.oof.season.unique()) - 1, key="slate")
    sl = P.slate(int(s))
    mx = len(sl) * (len(sl) + 1) / 2
    st.write(f"Out-of-fold picks (model trained **without** {s}). Winners right: **{int(sl.correct.sum())}/{len(sl)}**, "
             f"pick'em score **{int(sl.points.sum())}/{int(mx)}** ({sl.points.sum()/mx*100:.0f}%).")
    out = sl.sort_values("confidence_pts", ascending=False).copy()
    out["win_prob"] = (out.win_prob * 100).round(0).astype(int).astype(str) + "%"
    out["pred_margin"] = out.pred_margin.round(1)
    st.dataframe(out, hide_index=True, width="stretch")

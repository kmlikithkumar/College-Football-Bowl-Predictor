"""Command-line interface.

  python src/predict.py matchup 2018 "Alabama" "Oklahoma"     # predict any two FBS teams from a given season
  python src/predict.py slate 2019                            # pick'em sheet for a season (out-of-fold predictions)
"""
import sys
import pandas as pd
from predictor import BowlPredictor


def main(argv):
    need = {"matchup": 4, "slate": 2}
    if not argv or argv[0] not in need or len(argv) < need[argv[0]]:
        print(__doc__); return 1
    P = BowlPredictor()
    if argv[0] == "matchup":
        r = P.matchup(int(argv[1]), argv[2], argv[3])
        print(f"\n{r['season']} regular-season form:  {r['team1']}  vs  {r['team2']}")
        print(f"  Predicted winner : {r['winner']}   (win probability {r['win_prob']*100:.0f}%)")
        print(f"  Predicted margin : {r['winner']} by {abs(r['margin']):.1f} points")
        print("  Biggest drivers (points of margin toward", r["team1"], "):")
        for k, v in r["drivers"].items(): print(f"     {k:20s} {v:+.2f}")
    else:
        s = P.slate(int(argv[1]))
        pd.set_option("display.width", 200, "display.max_colwidth", 48)
        print(s.sort_values("confidence_pts", ascending=False).round({"pred_margin": 1, "win_prob": 2}).to_string(index=False))
        mx = len(s) * (len(s) + 1) / 2
        print(f"\nPick'em score: {s.points.sum()} / {int(mx)} ({s.points.sum()/mx*100:.0f}% of max); "
              f"winners right: {s.correct.sum()}/{len(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

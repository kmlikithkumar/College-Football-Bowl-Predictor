"""Builds the 2-page PDF write-up -> outputs/Writeup_G29_BowlPredictor.pdf"""
import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, "..", "outputs")
res = pd.read_csv(os.path.join(O, "results.csv")).set_index("model")
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.2, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#2b6cb0"))
S = ParagraphStyle("s", parent=B, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#444444"))
bul = lambda t: Paragraph("• " + t, ParagraphStyle("bl", parent=B, leftIndent=10, firstLineIndent=-8, spaceAfter=1.5))

rows = [["Model", "Winner acc.", "Pick'em (% of max)", "Margin MAE"]]
for k in ["Coin flip (reference)", "Baseline: higher win %", "Baseline: higher Elo", "Linear regression (all)", "Ridge", "Lasso",
          "Logistic regression", "Random forest", "Gradient boosting", "Ensemble (Ridge+RF+GB)", "Ridge: scoring/Elo only",
          "Ridge: efficiency stats only"]:
    r = res.loc[k]
    rows.append([k, f"{r.accuracy*100:.1f}% ± {r.acc_se*100:.1f}", f"{r.pickem_pct_of_max*100:.1f}%",
                 "-" if pd.isna(r.margin_MAE) else f"{r.margin_MAE:.1f}"])
t = Table(rows, colWidths=[6.2*cm, 3.2*cm, 3.6*cm, 2.6*cm])
t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                       ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#e6f0fa")), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))

doc = SimpleDocTemplate(os.path.join(O, "Writeup_G29_BowlPredictor.pdf"), pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm,
                        topMargin=1.3*cm, bottomMargin=1.2*cm, title="College Football Bowl Predictor")
el = [Paragraph("College Football Bowl Predictor", H1),
      Paragraph("UE24CS352A Machine Learning Mini-Project · Problem 88 · Team G-29: Likith Kumar K M (PES1UG24AM373), "
                "Uthkarsha (PES1UG24AM413) · Section G", S), Spacer(1, 3),
      Paragraph("1. Problem statement", H2),
      Paragraph("Predict the winner of each college football bowl game and attach a <b>confidence</b> to each pick, following "
                "Stanford CS229 (2015) project 101. Confidence matters because bowl pick'em contests rank your picks: a correct pick "
                "earns its confidence rank as points, so a good model must know <i>which</i> games it is sure about.", B),
      Paragraph("2. Dataset", H2),
      Paragraph("The original blog-sourced data is not available, so we rebuilt it from public play-by-play data "
                "(<i>sportsdataverse/cfbfastR-data</i>, GitHub): 16 seasons (2005-2020), ~12.6k games, ~25k team-games, aggregated "
                "from ~2M plays. We keep <b>566 bowl/postseason games</b>, of which 552 have two teams with ≥8 regular-season games. "
                "Per team-season we compute 36 features from <b>regular-season games only</b> (no leakage): scoring (PPG, PAPG), offensive "
                "and defensive yards/carry and yards/attempt, completion, sack and interception rates, EPA/play, success rate, havoc rate, "
                "turnover margin, penalties, punt average, FG%, and an end-of-season margin-aware <b>Elo rating</b> (K=20, home edge 55, "
                "1/3 regression to the mean each year). A game is the difference of the two teams' features; the label is the score margin.", B),
      Paragraph("3. Approach", H2),
      bul("<b>Symmetric formulation:</b> x = f(team1) − f(team2). Linear models have no intercept and are trained on randomly oriented pairs, "
          "so swapping teams exactly flips the prediction; tree models train on both orientations and are antisymmetrised."),
      bul("<b>Models:</b> OLS, Ridge, Lasso, logistic regression, random forest, gradient boosting, and an Ensemble, versus baselines "
          "(higher win %, higher Elo) and ablations (scoring/Elo features only; efficiency stats only)."),
      bul("<b>Validation:</b> leave-one-season-out CV (16 folds) with regularisation tuned by nested season-wise CV inside each training "
          "set. Metrics: winner accuracy, pick'em score (% of maximum N(N+1)/2 per season), margin MAE."),
      bul("<b>Confidence:</b> |predicted margin|, mapped to a win probability by a one-parameter logistic calibration fit on out-of-fold predictions."),
      Paragraph("4. Implementation", H2),
      Paragraph("Python (pandas, scikit-learn, matplotlib, Streamlit). <font face='Courier'>build_dataset.py</font> → "
                "<font face='Courier'>features.py</font> → <font face='Courier'>models.py/train.py</font> → "
                "<font face='Courier'>predictor.py</font> with a CLI (<font face='Courier'>predict.py</font>) and a Streamlit demo "
                "(matchup tab with per-feature explanation; season pick'em sheet). Full pipeline runs in ~3 minutes; the README has setup steps.", B),
      Paragraph("5. Results (552 bowl games, leave-one-season-out)", H2), t, Spacer(1, 3),
      Image(os.path.join(O, "figures", "model_comparison.png"), width=15.2*cm, height=6.1*cm)]
el += [Paragraph("6. Conclusions", H2),
       bul("<b>Ridge reaches 62.9% winner accuracy</b> (SE ≈ 2.1 pts) against 58.2% for the Elo baseline and 53.2% for win %. "
           "The gain over Elo is statistically suggestive (exact sign test, p = 0.029) but is one of five comparisons, so it is not conclusive."),
       bul("<b>Simple beats flexible.</b> With only ~550 games and 36 correlated features, regularised linear models match or beat random "
           "forest (61.1%), boosting (58.7%) and the ensemble (61.1%). Plain OLS (61.1%) is below Ridge, which shows shrinkage helps."),
       bul("<b>Scoring/Elo features carry the signal.</b> Ridge on only PPG, PAPG, margin, win % and Elo scores 63.1% (67.6% pick'em); "
           "advanced efficiency stats alone score 57.8%. The extra play-by-play features did not add measurable accuracy."),
       bul("<b>Confidence is only partly reliable.</b> Accuracy by confidence quartile is 64%, 59%, 56%, 73% (lowest to highest): the top "
           "quartile is clearly best, but the curve is not monotone, so the win-probability numbers should be read as rough."),
       bul("<b>Context.</b> The CS229 report cites ~66% on 2000-2010 data with different features; our data, seasons and protocol differ, "
           "so the numbers are not directly comparable. Bowl games have opt-outs, transfers and motivation effects that season statistics cannot see "
           "(<i>future work</i>: roster availability, betting lines, upset detection)."),
       Paragraph("Code: private GitHub repository (shared with faculty/TAs). Data: sportsdataverse/cfbfastR-data (public).", S)]
doc.build(el)
print("pages built")

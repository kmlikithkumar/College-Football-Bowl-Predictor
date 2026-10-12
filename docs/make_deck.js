const pptxgen = require("pptxgenjs");
const F = "outputs/figures/";
const NAVY="1F3A5F", BLUE="2B6CB0", LIGHT="E6F0FA", GREY="555555";
const p = new pptxgen(); p.layout = "LAYOUT_16x9";
const title = (s, t) => s.addText(t, {x:0.5,y:0.3,w:9,h:0.7,fontFace:"Calibri",fontSize:28,bold:true,color:NAVY,isTextBox:true,margin:0});
const bullets = (s, items, o={}) => s.addText(items.map((t,i)=>({text:t,options:{bullet:true,breakLine:i<items.length-1,paraSpaceAfter:8}})),
  Object.assign({x:0.5,y:1.2,w:9,h:3.9,fontFace:"Calibri",fontSize:18,color:"222222",valign:"top",isTextBox:true,margin:0},o));

let s = p.addSlide(); s.background={color:NAVY};
s.addText("College Football Bowl Predictor",{x:0.6,y:1.5,w:8.8,h:1,fontFace:"Calibri",fontSize:40,bold:true,color:"FFFFFF",isTextBox:true,margin:0});
s.addText("Picking bowl winners with a confidence, from regular-season data",{x:0.6,y:2.5,w:8.8,h:0.6,fontFace:"Calibri",fontSize:20,color:"BFD7F2",isTextBox:true,margin:0});
s.addText("UE24CS352A ML Mini-Project · Problem 88 · Team G-29\nLikith Kumar K M · Uthkarsha",{x:0.6,y:3.8,w:8.8,h:0.9,fontFace:"Calibri",fontSize:16,color:"FFFFFF",isTextBox:true,margin:0});

s = p.addSlide(); title(s,"Problem");
bullets(s,["Predict the winner of every bowl game, plus how confident we are (CS229 2015, project 101)",
 "Bowl pick'em: rank your picks by confidence; a correct pick earns its rank as points",
 "So we need two things: accurate winners and trustworthy confidence",
 "Challenge: only ~35 bowl games a year, and the original data is no longer available"]);

s = p.addSlide(); title(s,"Dataset: rebuilt from play-by-play");
bullets(s,["Source: sportsdataverse/cfbfastR-data (public GitHub), seasons 2005-2020",
 "~12.6k games, ~25k team-games, aggregated from ~2M plays",
 "566 bowl/postseason games (552 usable) are our samples",
 "36 features per team from the regular season only, so no leakage: scoring, yards/carry and /attempt, sack and INT rates, EPA, success and havoc rate, turnovers, penalties, punting, FG%, Elo"],{w:5.4});
s.addShape(p.ShapeType.rect,{x:6.2,y:1.3,w:3.3,h:3.2,fill:{color:LIGHT},line:{color:LIGHT}});
s.addText([{text:"552",options:{fontSize:44,bold:true,color:BLUE,breakLine:true}},{text:"bowl games",options:{fontSize:16,color:GREY,breakLine:true}},
 {text:"36",options:{fontSize:44,bold:true,color:BLUE,breakLine:true}},{text:"features per team",options:{fontSize:16,color:GREY}}],
 {x:6.2,y:1.3,w:3.3,h:3.2,align:"center",valign:"middle",fontFace:"Calibri",isTextBox:true});

s = p.addSlide(); title(s,"Approach");
bullets(s,["Game = features(team 1) − features(team 2); target = score margin",
 "Linear models have no intercept, so swapping teams exactly flips the prediction",
 "8 models: OLS, Ridge, Lasso, logistic, random forest, gradient boosting, ensemble, and baselines (win %, Elo)",
 "Leave-one-season-out CV over 16 seasons; regularisation tuned inside each training fold",
 "Confidence = |predicted margin|, calibrated to a win probability on out-of-fold predictions"]);

s = p.addSlide(); title(s,"Results: Ridge beats the baselines");
s.addImage({path:F+"model_comparison.png",x:0.4,y:1.1,w:6.4,h:2.56});
bullets(s,["Ridge: 62.9% (SE 2.1)","Elo baseline: 58.2%","Win % baseline: 53.2%","Ridge vs Elo: sign test p = 0.029"],{x:7,y:1.2,w:2.7,h:2.6,fontSize:15});
s.addText("Flexible models (RF 61.1%, boosting 58.7%) did not beat regularised linear ones with ~550 games.",{x:0.5,y:4.0,w:9,h:0.9,fontFace:"Calibri",fontSize:16,italic:true,color:GREY,isTextBox:true,margin:0});

s = p.addSlide(); title(s,"What carries the signal, and the confidence");
s.addImage({path:F+"ridge_coefficients.png",x:0.3,y:1.1,w:4.6,h:3.22});
s.addImage({path:F+"confidence_calibration.png",x:5.1,y:1.1,w:4.5,h:3.29});
s.addText("Scoring + Elo features alone: 63.1%. Efficiency stats alone: 57.8%. Top-confidence quartile: 73%, but quartiles are not monotone (64/59/56/73), so probabilities are rough.",
 {x:0.5,y:4.5,w:9,h:0.8,fontFace:"Calibri",fontSize:14,color:GREY,isTextBox:true,margin:0});

s = p.addSlide(); title(s,"Live demo");
bullets(s,["streamlit run src/app.py","Matchup tab: any two teams from a season, with predicted winner, margin, probability and the features driving it",
 "Pick'em tab: out-of-fold picks for a season with confidence ranks and score",
 "CLI: python src/predict.py slate 2019"]);

s = p.addSlide(); title(s,"Conclusions and limits");
bullets(s,["~63% winner accuracy vs 58% Elo and 53% win %: a real but modest gain",
 "Simple, regularised models win on small data",
 "Confidence helps at the top end only",
 "Limits: opt-outs, transfers and motivation are invisible to season stats; Ridge-vs-Elo p-value is one of five tests",
 "Next: betting lines, roster availability, upset detection"]);
p.writeFile({fileName:"outputs/Slides_G29_BowlPredictor.pptx"}).then(()=>console.log("deck ok"));

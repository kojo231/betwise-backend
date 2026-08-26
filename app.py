import os
from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
from datetime import date

app = Flask(__name__)
CORS(app)

# --- TEAM NAME MAPPING ---
TEAM_MAP = {
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Nottingham Forest": "Nottm Forest",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd",
    "Ipswich Town": "Ipswich",
    "Atletico Madrid": "Atletico Madrid",
    "Athletic Club": "Ath Bilbao",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "AC Milan": "AC Milan",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "Juventus FC": "Juventus",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Paris Saint-Germain": "PSG",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting",
}

def map_team(name):
    if not name:
        return name
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    for api_name, dataset_name in TEAM_MAP.items():
        if api_name.lower() in name.lower() or name.lower() in api_name.lower():
            return dataset_name
    return name


# --- TRAIN MODEL ---
def train_model():
    try:
        df = pd.read_csv("Master_BetWise_Features.csv", low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, {}, {}, {}

    df = df.dropna(subset=["B365H", "B365D", "B365A", "FTR"])
    df["ProbH"] = 1 / df["B365H"]
    df["ProbD"] = 1 / df["B365D"]
    df["ProbA"] = 1 / df["B365A"]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    home_form = np.zeros(len(df))
    away_form = np.zeros(len(df))
    home_gs = np.zeros(len(df))
    away_gs = np.zeros(len(df))
    home_gc = np.zeros(len(df))
    away_gc = np.zeros(len(df))
    home_streak = np.zeros(len(df))
    away_streak = np.zeros(len(df))
    home_history = {}
    away_history = {}
    team_streak = {}

    for i, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        h_hist = home_history.get(ht, [])[-5:]
        a_hist = away_history.get(at, [])[-5:]

        if h_hist:
            home_form[i] = sum(x[0] for x in h_hist)
            home_gs[i] = sum(x[1] for x in h_hist)
            home_gc[i] = sum(x[2] for x in h_hist)

        if a_hist:
            away_form[i] = sum(x[0] for x in a_hist)
            away_gs[i] = sum(x[1] for x in a_hist)
            away_gc[i] = sum(x[2] for x in a_hist)

        home_streak[i] = team_streak.get(ht, 0)
        away_streak[i] = team_streak.get(at, 0)

        if row["FTR"] == "H": h_pts, a_pts = 3, 0
        elif row["FTR"] == "D": h_pts, a_pts = 1, 1
        else: h_pts, a_pts = 0, 3

        if ht not in home_history: home_history[ht] = []
        if at not in away_history: away_history[at] = []
        home_history[ht].append((h_pts, row["FTHG"], row["FTAG"]))
        away_history[at].append((a_pts, row["FTAG"], row["FTHG"]))

        if row["FTR"] == "H":
            team_streak[ht] = max(0, team_streak.get(ht, 0)) + 1
            team_streak[at] = min(0, team_streak.get(at, 0)) - 1
        elif row["FTR"] == "A":
            team_streak[ht] = min(0, team_streak.get(ht, 0)) - 1
            team_streak[at] = max(0, team_streak.get(at, 0)) + 1
        else:
            team_streak[ht] = 0
            team_streak[at] = 0

    df["HomeForm"] = home_form
    df["AwayForm"] = away_form
    df["HomeGS"] = home_gs
    df["AwayGS"] = away_gs
    df["HomeGC"] = home_gc
    df["AwayGC"] = away_gc
    df["HomeStreak"] = home_streak
    df["AwayStreak"] = away_streak
    df["FormDiff"] = df["HomeForm"] - df["AwayForm"]
    df["MomentumDiff"] = df["HomeStreak"] - df["AwayStreak"]

    features = [
        "ProbH", "ProbD", "ProbA",
        "HomeForm", "AwayForm", "FormDiff",
        "HomeGS", "AwayGS", "HomeGC", "AwayGC",
        "HomeStreak", "AwayStreak", "MomentumDiff"
    ]

    X = df[features].fillna(0)
    y = df["FTR"]
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X, y)
    return model, home_history, away_history, team_streak


print("Training model...")
model, home_history, away_history, team_streak = train_model()
print("Model ready!")


# --- PREDICTION ---
def predict_match(home_team, away_team, h_odds=2.0, d_odds=3.4, a_odds=4.0):
    """
    Predict match outcome.
    
    Key improvement: When a team has little or no current season data,
    we rely more heavily on bookmaker odds rather than defaulting
    to home bias. Bookmakers already have current information built
    into their odds.
    """
    def get_stats(team, history):
        hist = history.get(team, [])[-5:]
        return hist, len(hist)

    h_hist, h_data_count = get_stats(home_team, home_history)
    a_hist, a_data_count = get_stats(away_team, away_history)

    # How much data do we actually have?
    # 0 games = no data, 5 games = full confidence
    h_confidence = h_data_count / 5.0
    a_confidence = a_data_count / 5.0
    data_confidence = (h_confidence + a_confidence) / 2.0

    # Calculate stats from available history
    if h_hist:
        h_form = sum(x[0] for x in h_hist)
        h_gs = sum(x[1] for x in h_hist)
        h_gc = sum(x[2] for x in h_hist)
    else:
        # No data — use neutral values, not home-biased defaults
        h_form = 5.0  # neutral form
        h_gs = 1.3 * len(h_hist) if h_hist else 6.5
        h_gc = 1.3 * len(h_hist) if h_hist else 6.5

    if a_hist:
        a_form = sum(x[0] for x in a_hist)
        a_gs = sum(x[1] for x in a_hist)
        a_gc = sum(x[2] for x in a_hist)
    else:
        a_form = 5.0  # neutral form
        a_gs = 1.3 * len(a_hist) if a_hist else 6.5
        a_gc = 1.3 * len(a_hist) if a_hist else 6.5

    h_streak = team_streak.get(home_team, 0)
    a_streak = team_streak.get(away_team, 0)

    # Model prediction
    X = pd.DataFrame([{
        "ProbH": 1 / h_odds, "ProbD": 1 / d_odds, "ProbA": 1 / a_odds,
        "HomeForm": h_form, "AwayForm": a_form, "FormDiff": h_form - a_form,
        "HomeGS": h_gs, "AwayGS": a_gs, "HomeGC": h_gc, "AwayGC": a_gc,
        "HomeStreak": h_streak, "AwayStreak": a_streak,
        "MomentumDiff": h_streak - a_streak
    }])

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    model_result = {c: float(p) for c, p in zip(classes, proba)}

    model_h = model_result.get("H", 0.33)
    model_d = model_result.get("D", 0.33)
    model_a = model_result.get("A", 0.33)

    # Bookmaker implied probabilities (normalize to sum to 1)
    raw_h = 1 / h_odds
    raw_d = 1 / d_odds
    raw_a = 1 / a_odds
    total = raw_h + raw_d + raw_a
    bookie_h = raw_h / total
    bookie_d = raw_d / total
    bookie_a = raw_a / total

    # BLEND: When data is scarce, trust bookmakers more
    # data_confidence = 0 means no data → use bookmakers 100%
    # data_confidence = 1 means full data → use model 70%, bookmakers 30%
    max_model_weight = 0.65
    model_weight = data_confidence * max_model_weight
    bookie_weight = 1.0 - model_weight

    final_h = round((model_h * model_weight + bookie_h * bookie_weight) * 100, 1)
    final_d = round((model_d * model_weight + bookie_d * bookie_weight) * 100, 1)
    final_a = round((model_a * model_weight + bookie_a * bookie_weight) * 100, 1)

    # Normalize to 100%
    total_pct = final_h + final_d + final_a
    final_h = round(final_h / total_pct * 100, 1)
    final_d = round(final_d / total_pct * 100, 1)
    final_a = round(100 - final_h - final_d, 1)

    # Confidence level — honest about data limitations
    max_pct = max(final_h, final_d, final_a)
    if data_confidence < 0.3:
        conf_label = "Low"  # Honest: very little current data
    elif data_confidence < 0.7:
        conf_label = "Low" if max_pct < 50 else "Medium"
    else:
        conf_label = "High" if max_pct >= 60 else "Medium" if max_pct >= 50 else "Low"

    prediction = "H" if final_h == max(final_h, final_d, final_a) else \
                 "A" if final_a == max(final_h, final_d, final_a) else "D"

    # Goals prediction
    avg_goals = ((h_gs + a_gs) / max(len(h_hist) + len(a_hist), 1)) if (h_hist or a_hist) else 2.6

    def poisson_over(lam, threshold):
        prob_under = sum([(lam**k * math.exp(-lam)) / math.factorial(k)
                         for k in range(int(threshold) + 1)])
        return round((1 - prob_under) * 100, 1)

    over_under = {k: poisson_over(avg_goals, float(k) - 0.5)
                  for k in ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"]}

    def streak_label(s):
        if s >= 3: return f"{s} game win run"
        elif s <= -3: return f"{abs(s)} game losing run"
        elif s > 0: return f"{s} game win run"
        elif s < 0: return f"{abs(s)} game losing run"
        return "No clear streak"

    return {
        "home_pct": final_h,
        "draw_pct": final_d,
        "away_pct": final_a,
        "prediction": prediction,
        "confidence": round(max_pct, 1),
        "confidence_label": conf_label,
        "data_confidence": round(data_confidence * 100),
        "over_under": over_under,
        "double_chance": {
            "home_or_draw": round(final_h + final_d, 1),
            "away_or_draw": round(final_a + final_d, 1),
            "home_or_away": round(final_h + final_a, 1)
        },
        "home_to_score": round(min(90, max(35, (h_gs / max(len(h_hist), 1)) * 65)), 1),
        "away_to_score": round(min(85, max(30, (a_gs / max(len(a_hist), 1)) * 60)), 1),
        "home_streak": streak_label(h_streak),
        "away_streak": streak_label(a_streak),
        "home_avg_goals": round(h_gs / max(len(h_hist), 1), 1) if h_hist else 1.4,
        "away_avg_goals": round(a_gs / max(len(a_hist), 1), 1) if a_hist else 1.2,
    }


# --- ROUTES ---
@app.route("/api/fixtures", methods=["GET"])
def get_fixtures():
    try:
        api_key = os.environ.get("FOOTBALL_API_TOKEN", "")
        headers = {"X-Auth-Token": api_key}

        res = requests.get(
            "https://api.football-data.org/v4/matches?status=SCHEDULED",
            headers=headers,
            timeout=10
        )
        matches = res.json().get("matches", [])

        fixtures = []
        for m in matches:
            raw_home = m["homeTeam"]["shortName"]
            raw_away = m["awayTeam"]["shortName"]
            date_str = m["utcDate"][:10]
            time_str = m["utcDate"][11:16]
            league_name = m["competition"]["name"]

            # Get odds if available
            h_odds = 2.0
            d_odds = 3.4
            a_odds = 4.0
            
            home = map_team(raw_home)
            away = map_team(raw_away)

            pred = predict_match(home, away, h_odds, d_odds, a_odds)
            pred["home_team"] = raw_home
            pred["away_team"] = raw_away
            pred["date"] = date_str
            pred["time"] = time_str
            pred["competition"] = league_name
            fixtures.append(pred)

        print(f"Returning {len(fixtures)} fixtures")
        return jsonify({"fixtures": fixtures, "source": "live"})

    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"fixtures": [], "source": "error", "error": str(e)})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "BetWise API running", "model": "loaded"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
import os
from flask import Flask, jsonify, request
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
    def get_stats(team, history):
        hist = history.get(team, [])[-5:]
        if not hist:
            return 7, 2.0, 1.5
        return sum(x[0] for x in hist), sum(x[1] for x in hist), sum(x[2] for x in hist)

    h_form, h_gs, h_gc = get_stats(home_team, home_history)
    a_form, a_gs, a_gc = get_stats(away_team, away_history)
    h_streak = team_streak.get(home_team, 0)
    a_streak = team_streak.get(away_team, 0)

    X = pd.DataFrame([{
        "ProbH": 1 / h_odds, "ProbD": 1 / d_odds, "ProbA": 1 / a_odds,
        "HomeForm": h_form, "AwayForm": a_form, "FormDiff": h_form - a_form,
        "HomeGS": h_gs, "AwayGS": a_gs, "HomeGC": h_gc, "AwayGC": a_gc,
        "HomeStreak": h_streak, "AwayStreak": a_streak,
        "MomentumDiff": h_streak - a_streak
    }])

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    result = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    home_pct = result.get("H", 33)
    draw_pct = result.get("D", 33)
    away_pct = result.get("A", 33)
    confidence = round(max(home_pct, draw_pct, away_pct), 1)
    conf_label = "High" if confidence >= 60 else "Medium" if confidence >= 50 else "Low"
    prediction = "H" if home_pct == max(home_pct, draw_pct, away_pct) else \
                 "A" if away_pct == max(home_pct, draw_pct, away_pct) else "D"

    avg_goals = (h_gs + a_gs) / 5 if h_gs + a_gs > 0 else 2.5

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
        "home_pct": home_pct, "draw_pct": draw_pct, "away_pct": away_pct,
        "prediction": prediction, "confidence": confidence,
        "confidence_label": conf_label, "over_under": over_under,
        "double_chance": {
            "home_or_draw": round(home_pct + draw_pct, 1),
            "away_or_draw": round(away_pct + draw_pct, 1),
            "home_or_away": round(home_pct + away_pct, 1)
        },
        "home_to_score": round(min(95, max(30, (h_gs / 5) * 30)), 1),
        "away_to_score": round(min(95, max(30, (a_gs / 5) * 30)), 1),
        "home_streak": streak_label(h_streak),
        "away_streak": streak_label(a_streak),
        "home_avg_goals": round(h_gs / 5, 1) if h_gs > 0 else 1.5,
        "away_avg_goals": round(a_gs / 5, 1) if a_gs > 0 else 1.2,
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

            home = map_team(raw_home)
            away = map_team(raw_away)

            pred = predict_match(home, away)
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
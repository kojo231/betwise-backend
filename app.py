import os
import json
from datetime import date, datetime
from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math

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

    # Use class_weight balanced to reduce home bias
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=10,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model, home_history, away_history, team_streak


print("Training model...")
model, home_history, away_history, team_streak = train_model()
print("Model ready!")


# --- PREDICTION ---
def predict_match(home_team, away_team, h_odds=None, d_odds=None, a_odds=None):
    def get_stats(team, history):
        hist = history.get(team, [])[-5:]
        if not hist:
            return 0, 0, 0
        return sum(x[0] for x in hist), sum(x[1] for x in hist), sum(x[2] for x in hist)

    h_form, h_gs, h_gc = get_stats(home_team, home_history)
    a_form, a_gs, a_gc = get_stats(away_team, away_history)
    h_streak = team_streak.get(home_team, 0)
    a_streak = team_streak.get(away_team, 0)

    # Use odds if provided otherwise use neutral odds
    # Neutral odds = equal probability for all outcomes
    h_odds = h_odds or 3.0
    d_odds = d_odds or 3.0
    a_odds = a_odds or 3.0

    X = pd.DataFrame([{
        "ProbH": 1 / h_odds,
        "ProbD": 1 / d_odds,
        "ProbA": 1 / a_odds,
        "HomeForm": h_form,
        "AwayForm": a_form,
        "FormDiff": h_form - a_form,
        "HomeGS": h_gs,
        "AwayGS": a_gs,
        "HomeGC": h_gc,
        "AwayGC": a_gc,
        "HomeStreak": h_streak,
        "AwayStreak": a_streak,
        "MomentumDiff": h_streak - a_streak
    }])

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    result = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    home_pct = result.get("H", 33.3)
    draw_pct = result.get("D", 33.3)
    away_pct = result.get("A", 33.3)

    max_pct = max(home_pct, draw_pct, away_pct)
    prediction = "H" if home_pct == max_pct else "A" if away_pct == max_pct else "D"

    conf_label = "High" if max_pct >= 60 else "Medium" if max_pct >= 50 else "Low"

    avg_goals = (h_gs + a_gs) / 10 if (h_gs + a_gs) > 0 else 2.6

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
        "home_pct": home_pct,
        "draw_pct": draw_pct,
        "away_pct": away_pct,
        "prediction": prediction,
        "confidence": round(max_pct, 1),
        "confidence_label": conf_label,
        "over_under": over_under,
        "double_chance": {
            "home_or_draw": round(home_pct + draw_pct, 1),
            "away_or_draw": round(away_pct + draw_pct, 1),
            "home_or_away": round(home_pct + away_pct, 1)
        },
        "home_to_score": round(min(90, max(30, (h_gs / 5) * 60)), 1),
        "away_to_score": round(min(85, max(25, (a_gs / 5) * 55)), 1),
        "home_streak": streak_label(h_streak),
        "away_streak": streak_label(a_streak),
        "home_avg_goals": round(h_gs / 5, 1) if h_gs > 0 else 1.3,
        "away_avg_goals": round(a_gs / 5, 1) if a_gs > 0 else 1.1,
    }


# --- PREDICTION HISTORY ---
HISTORY_FILE = "prediction_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_to_history(fixture):
    history = load_history()
    key = f"{fixture['date']}_{fixture['home_team']}_{fixture['away_team']}"
    existing_keys = [f"{h['date']}_{h['home_team']}_{h['away_team']}" for h in history]
    if key not in existing_keys:
        history.append({
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "competition": fixture["competition"],
            "prediction": fixture["prediction"],
            "home_pct": fixture["home_pct"],
            "draw_pct": fixture["draw_pct"],
            "away_pct": fixture["away_pct"],
            "confidence_label": fixture["confidence_label"],
            "actual_result": None,
            "correct": None,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        history = history[-200:]
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)


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

            save_to_history(pred)
            fixtures.append(pred)

        print(f"Returning {len(fixtures)} fixtures")
        return jsonify({"fixtures": fixtures, "source": "live"})

    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"fixtures": [], "source": "error", "error": str(e)})


@app.route("/api/history", methods=["GET"])
def get_history():
    history = load_history()
    return jsonify({"history": list(reversed(history))})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "BetWise API running", "model": "loaded"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
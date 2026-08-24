"""
BetWise Model v2 - With Validated Strategies
Tests accuracy before and after adding:
1. Losing streak bounce back (home teams, 3-4 games)
2. Goal difference trend (rolling pre-match average)

HOW TO USE:
Run: python model_v2.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
df = pd.read_csv('Master_BetWise_Features.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])

print(f"Total matches: {len(df)}")

# --- Calculate additional strategy features ---
print("Calculating strategy features...")

team_streak = {}
team_gd = {}
home_streak_vals = []
away_streak_vals = []
home_gd_trend = []
away_gd_trend = []

for idx, row in df.iterrows():
    ht = row['HomeTeam']
    at = row['AwayTeam']

    # Streaks
    h_streak = team_streak.get(ht, 0)
    a_streak = team_streak.get(at, 0)
    home_streak_vals.append(h_streak)
    away_streak_vals.append(a_streak)

    # Goal difference trend (last 5 matches)
    h_gd_hist = team_gd.get(ht, [])[-5:]
    a_gd_hist = team_gd.get(at, [])[-5:]
    home_gd_trend.append(np.mean(h_gd_hist) if h_gd_hist else 0)
    away_gd_trend.append(np.mean(a_gd_hist) if a_gd_hist else 0)

    # Update after recording
    ftr = row['FTR']
    fthg = row['FTHG'] if not pd.isna(row.get('FTHG', np.nan)) else 0
    ftag = row['FTAG'] if not pd.isna(row.get('FTAG', np.nan)) else 0

    if ftr == 'H': h_pts, a_pts = 3, 0
    elif ftr == 'D': h_pts, a_pts = 1, 1
    else: h_pts, a_pts = 0, 3

    # Update streaks
    if ftr == 'H':
        team_streak[ht] = max(0, team_streak.get(ht, 0)) + 1
        team_streak[at] = min(0, team_streak.get(at, 0)) - 1
    elif ftr == 'A':
        team_streak[ht] = min(0, team_streak.get(ht, 0)) - 1
        team_streak[at] = max(0, team_streak.get(at, 0)) + 1
    else:
        team_streak[ht] = 0
        team_streak[at] = 0

    # Update GD history
    if ht not in team_gd: team_gd[ht] = []
    if at not in team_gd: team_gd[at] = []
    team_gd[ht].append(fthg - ftag)
    team_gd[at].append(ftag - fthg)

df['HomeStreakCalc'] = home_streak_vals
df['AwayStreakCalc'] = away_streak_vals
df['HomeGDTrend'] = home_gd_trend
df['AwayGDTrend'] = away_gd_trend

# Losing streak bounce back flag
# Home team on 3-4 game losing streak
df['HomeBounceBack'] = ((df['HomeStreakCalc'] == -3) | 
                         (df['HomeStreakCalc'] == -4)).astype(int)

# GD trend difference
df['GDTrendDiff'] = df['HomeGDTrend'] - df['AwayGDTrend']

# Bookmaker probabilities
df['ProbH'] = 1 / df['B365H']
df['ProbD'] = 1 / df['B365D']
df['ProbA'] = 1 / df['B365A']

# Train/test split — test on 2024/25 season
test_start = pd.Timestamp('2024-08-01')
train = df[df['Date'] < test_start].copy()
test = df[df['Date'] >= test_start].copy()

print(f"Training: {len(train)} | Testing: {len(test)}")

# --- BASELINE MODEL (no new strategies) ---
print("\n=== BASELINE MODEL ===")
base_features = [
    'ProbH', 'ProbD', 'ProbA',
    'HomeForm', 'AwayForm', 'FormDiff',
    'HomeGS', 'AwayGS', 'HomeGC', 'AwayGC',
    'HomeStreak', 'AwayStreak', 'MomentumDiff',
    'HomeAvgGoals', 'AwayAvgGoals',
]

# Only use features that exist
base_features = [f for f in base_features if f in df.columns]

X_train_base = train[base_features].fillna(0)
y_train = train['FTR']
X_test_base = test[base_features].fillna(0)
y_test = test['FTR']

model_base = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model_base.fit(X_train_base, y_train)
y_pred_base = model_base.predict(X_test_base)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"🎯 Baseline Accuracy: {acc_base*100:.1f}%")

# --- STRATEGY MODEL (with new features) ---
print("\n=== STRATEGY MODEL (with validated strategies) ===")
strategy_features = base_features + [
    'HomeBounceBack',
    'HomeGDTrend', 'AwayGDTrend', 'GDTrendDiff',
    'HomeStreakCalc', 'AwayStreakCalc',
]

strategy_features = [f for f in strategy_features if f in df.columns]

X_train_strat = train[strategy_features].fillna(0)
X_test_strat = test[strategy_features].fillna(0)

model_strat = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model_strat.fit(X_train_strat, y_train)
y_pred_strat = model_strat.predict(X_test_strat)
acc_strat = accuracy_score(y_test, y_pred_strat)
print(f"🎯 Strategy Model Accuracy: {acc_strat*100:.1f}%")

# --- COMPARISON ---
print("\n=== HONEST COMPARISON ===")
print(f"Baseline:        {acc_base*100:.1f}%")
print(f"With strategies: {acc_strat*100:.1f}%")
diff = (acc_strat - acc_base) * 100

if diff > 0:
    print(f"✅ Improvement: +{diff:.1f} percentage points")
    print("Strategies genuinely help — worth keeping")
elif diff == 0:
    print("➡️ No change in accuracy")
    print("Strategies add no value — consider removing")
else:
    print(f"⚠️ Accuracy dropped: {diff:.1f} percentage points")
    print("Strategies hurt accuracy — do not use")

print("\n=== DETAILED BREAKDOWN (Strategy Model) ===")
print(classification_report(y_test, y_pred_strat,
      target_names=['Away Win', 'Draw', 'Home Win']))

print("\n=== FEATURE IMPORTANCE ===")
imp = pd.DataFrame({
    'Feature': strategy_features,
    'Importance': model_strat.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

for _, row in imp.iterrows():
    bar = '█' * int(row['Importance'] * 60)
    print(f"{row['Feature']:<22} {bar} {row['Importance']*100:.1f}%")

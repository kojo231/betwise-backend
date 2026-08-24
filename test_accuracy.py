"""
BetWise Model Accuracy Test
Compares prediction accuracy between:
- Old model: Premier League only
- New model: All 6 leagues combined

Uses 2024/25 season as test data (unseen by training)
Trains on everything before 2024/25

HOW TO USE:
Run: python test_accuracy.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES_FILE = "Master_BetWise_Features.csv"

FEATURES = [
    "ProbH", "ProbD", "ProbA",
    "HomeForm", "AwayForm", "FormDiff",
    "HomeGS", "AwayGS", "HomeGC", "AwayGC",
    "HomeStreak", "AwayStreak", "MomentumDiff",
    "HomeAvgGoals", "AwayAvgGoals",
]

def test_accuracy():
    print("=== BetWise Accuracy Test ===\n")

    df = pd.read_csv(FEATURES_FILE, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.dropna(subset=['FTR', 'B365H', 'B365D', 'B365A'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Total matches available: {len(df)}")

    # Define test period — 2024/25 season
    test_start = pd.Timestamp('2024-08-01')

    train = df[df['Date'] < test_start].copy()
    test = df[df['Date'] >= test_start].copy()

    print(f"Training matches: {len(train)}")
    print(f"Testing matches: {len(test)}")

    # --- TEST 1: Premier League Only ---
    print("\n=== TEST 1: Premier League Only (Old Approach) ===")
    pl_train = train[train['League'] == 'England Premier League']
    pl_test = test[test['League'] == 'England Premier League']

    X_train = pl_train[FEATURES].fillna(0)
    y_train = pl_train['FTR']
    X_test = pl_test[FEATURES].fillna(0)
    y_test = pl_test['FTR']

    model1 = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)

    print(f"Training matches: {len(pl_train)}")
    print(f"Testing matches: {len(pl_test)}")
    print(f"🎯 Premier League Only Accuracy: {acc1*100:.1f}%")
    print(classification_report(y_test, y_pred1,
          target_names=['Away Win', 'Draw', 'Home Win']))

    # --- TEST 2: All Leagues Combined ---
    print("\n=== TEST 2: All 6 Leagues Combined (New Approach) ===")
    X_train2 = train[FEATURES].fillna(0)
    y_train2 = train['FTR']
    X_test2 = test[FEATURES].fillna(0)
    y_test2 = test['FTR']

    model2 = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model2.fit(X_train2, y_train2)
    y_pred2 = model2.predict(X_test2)
    acc2 = accuracy_score(y_test2, y_pred2)

    print(f"Training matches: {len(train)}")
    print(f"Testing matches: {len(test)}")
    print(f"🎯 All Leagues Accuracy: {acc2*100:.1f}%")
    print(classification_report(y_test2, y_pred2,
          target_names=['Away Win', 'Draw', 'Home Win']))

    # --- TEST 3: All Leagues but test only on Premier League ---
    print("\n=== TEST 3: Trained on All Leagues, Tested on Premier League ===")
    y_pred3 = model2.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred3)
    print(f"🎯 Accuracy on Premier League: {acc3*100:.1f}%")

    # --- Per League Accuracy ---
    print("\n=== ACCURACY PER LEAGUE (New Model) ===")
    for league in test['League'].unique():
        league_test = test[test['League'] == league]
        if len(league_test) < 10:
            continue
        X_l = league_test[FEATURES].fillna(0)
        y_l = league_test['FTR']
        y_p = model2.predict(X_l)
        acc_l = accuracy_score(y_l, y_p)
        print(f"  {league}: {acc_l*100:.1f}% ({len(league_test)} matches)")

    # --- Honest Summary ---
    print("\n=== HONEST SUMMARY ===")
    print(f"Premier League only model: {acc1*100:.1f}%")
    print(f"All leagues combined model: {acc2*100:.1f}%")
    print(f"All leagues model on PL only: {acc3*100:.1f}%")

    if acc2 > acc1:
        diff = (acc2 - acc1) * 100
        print(f"\n✅ New model is BETTER by {diff:.1f} percentage points")
        print("Recommendation: Update app.py to use new dataset")
    elif acc2 == acc1:
        print(f"\n➡️ Models perform equally")
        print("Recommendation: Use new dataset for broader league coverage")
    else:
        diff = (acc1 - acc2) * 100
        print(f"\n⚠️ New model is WORSE by {diff:.1f} percentage points")
        print("Recommendation: Investigate before updating app.py")

if __name__ == "__main__":
    test_accuracy()

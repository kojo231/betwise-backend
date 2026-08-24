"""
BetWise Strategy Test
Tests the losing streak bounce back theory against real data.

HOW TO USE:
Run: python test_losing_streak.py
"""

import pandas as pd
import numpy as np

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Master_BetWise_Features.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna(subset=['FTR', 'HomeTeam', 'AwayTeam'])

print(f"Total matches: {len(df)}")

# Track streaks and record results
team_streak = {}
results = []

for idx, row in df.iterrows():
    ht = row['HomeTeam']
    at = row['AwayTeam']
    ftr = row['FTR']
    league = row['League']

    h_streak = team_streak.get(ht, 0)
    a_streak = team_streak.get(at, 0)

    # Record if home team on 3+ losing streak
    if h_streak <= -3:
        results.append({
            'team': ht,
            'role': 'home',
            'streak': h_streak,
            'won': ftr == 'H',
            'drew': ftr == 'D',
            'won_or_drew': ftr in ['H', 'D'],
            'league': league
        })

    # Record if away team on 3+ losing streak
    if a_streak <= -3:
        results.append({
            'team': at,
            'role': 'away',
            'streak': a_streak,
            'won': ftr == 'A',
            'drew': ftr == 'D',
            'won_or_drew': ftr in ['A', 'D'],
            'league': league
        })

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

results_df = pd.DataFrame(results)
total = len(results_df)

print(f"\n=== LOSING STREAK BOUNCE BACK TEST ===")
print(f"Matches where team was on 3+ game losing streak: {total}")

won_or_drew = results_df['won_or_drew'].sum()
won = results_df['won'].sum()
drew = results_df['drew'].sum()

print(f"\n=== OVERALL RESULTS ===")
print(f"Won or Drew: {won_or_drew}/{total} = {won_or_drew/total*100:.1f}%")
print(f"Won only:    {won}/{total} = {won/total*100:.1f}%")
print(f"Drew only:   {drew}/{total} = {drew/total*100:.1f}%")
print(f"Lost again:  {total-won_or_drew}/{total} = {(total-won_or_drew)/total*100:.1f}%")

print(f"\n=== BY STREAK LENGTH ===")
for s in [-3, -4, -5, -6, -7]:
    sub = results_df[results_df['streak'] == s]
    if len(sub) >= 10:
        pct = sub['won_or_drew'].mean() * 100
        print(f"  {abs(s)} game losing streak: {pct:.1f}% won or drew ({len(sub)} matches)")

print(f"\n=== HOME vs AWAY ===")
home = results_df[results_df['role'] == 'home']
away = results_df[results_df['role'] == 'away']
print(f"  Home team on losing streak: {home['won_or_drew'].mean()*100:.1f}% won or drew ({len(home)} matches)")
print(f"  Away team on losing streak: {away['won_or_drew'].mean()*100:.1f}% won or drew ({len(away)} matches)")

print(f"\n=== BY LEAGUE ===")
for league in results_df['league'].unique():
    sub = results_df[results_df['league'] == league]
    if len(sub) >= 20:
        pct = sub['won_or_drew'].mean() * 100
        print(f"  {league}: {pct:.1f}% won or drew ({len(sub)} matches)")

print(f"\n=== HONEST VERDICT ===")
overall_pct = won_or_drew/total*100
if overall_pct >= 70:
    print(f"✅ Strategy VALIDATED — {overall_pct:.1f}% won or drew")
    print("This is significantly above average and worth building into the model")
elif overall_pct >= 55:
    print(f"⚠️ Strategy PARTIALLY VALID — {overall_pct:.1f}% won or drew")
    print("Better than random but not as strong as claimed")
else:
    print(f"❌ Strategy NOT VALIDATED — {overall_pct:.1f}% won or drew")
    print("Data does not support this claim strongly enough")

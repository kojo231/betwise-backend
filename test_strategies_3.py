"""
BetWise Strategy Test 3
Tests:
1. Goal difference trends
2. Shots dominance (HST)
3. Odds movement (opening vs closing)

HOW TO USE:
Run: python test_strategies_3.py
"""

import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv('Master_BetWise_Features.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna(subset=['FTR', 'HomeTeam', 'AwayTeam'])
print(f"Total matches: {len(df)}")

# ================================================
# TEST 1: GOAL DIFFERENCE TRENDS
# Teams scoring more than they concede consistently
# ================================================
print("\n=== TEST 1: GOAL DIFFERENCE TRENDS ===")

team_gd = {}
home_gd_trend = []
away_gd_trend = []

for idx, row in df.iterrows():
    ht = row['HomeTeam']
    at = row['AwayTeam']

    h_hist = team_gd.get(ht, [])[-5:]
    a_hist = team_gd.get(at, [])[-5:]

    h_gd = sum(h_hist) / len(h_hist) if h_hist else 0
    a_gd = sum(a_hist) / len(a_hist) if a_hist else 0

    home_gd_trend.append(h_gd)
    away_gd_trend.append(a_gd)

    fthg = row['FTHG'] if not pd.isna(row.get('FTHG', np.nan)) else 0
    ftag = row['FTAG'] if not pd.isna(row.get('FTAG', np.nan)) else 0

    if ht not in team_gd: team_gd[ht] = []
    if at not in team_gd: team_gd[at] = []
    team_gd[ht].append(fthg - ftag)
    team_gd[at].append(ftag - fthg)

df['HomeGDTrend'] = home_gd_trend
df['AwayGDTrend'] = away_gd_trend
df['GDTrendDiff'] = df['HomeGDTrend'] - df['AwayGDTrend']

# Test: When home team has significantly better GD trend
strong_home = df[df['GDTrendDiff'] > 1.0].dropna(subset=['FTR'])
weak_home = df[df['GDTrendDiff'] < -1.0].dropna(subset=['FTR'])

if len(strong_home) > 0 and len(weak_home) > 0:
    strong_win = (strong_home['FTR'] == 'H').mean() * 100
    weak_win = (weak_home['FTR'] == 'H').mean() * 100
    print(f"Home team with strong GD trend (>1.0): wins {strong_win:.1f}% ({len(strong_home)} matches)")
    print(f"Home team with weak GD trend (<-1.0): wins {weak_win:.1f}% ({len(weak_home)} matches)")
    print(f"Difference: {abs(strong_win - weak_win):.1f} percentage points")
    if abs(strong_win - weak_win) >= 3:
        print("✅ Goal difference trend IS worth adding to model")
    else:
        print("⚠️ Goal difference trend has minimal impact")

# ================================================
# TEST 2: SHOTS ON TARGET DOMINANCE
# Teams with more shots on target win more
# ================================================
print("\n=== TEST 2: SHOTS ON TARGET DOMINANCE ===")

shots_df = df.dropna(subset=['HST', 'AST', 'FTR'])

if len(shots_df) > 0:
    shots_df = shots_df.copy()
    shots_df['ShotsDiff'] = shots_df['HST'] - shots_df['AST']

    home_dominant = shots_df[shots_df['ShotsDiff'] >= 3]
    away_dominant = shots_df[shots_df['ShotsDiff'] <= -3]
    even = shots_df[shots_df['ShotsDiff'].between(-1, 1)]

    print(f"Home team dominant in shots (HST diff >= 3):")
    if len(home_dominant) > 0:
        print(f"  Home win rate: {(home_dominant['FTR']=='H').mean()*100:.1f}% ({len(home_dominant)} matches)")

    print(f"Away team dominant in shots (HST diff <= -3):")
    if len(away_dominant) > 0:
        print(f"  Away win rate: {(away_dominant['FTR']=='A').mean()*100:.1f}% ({len(away_dominant)} matches)")

    print(f"Even shots:")
    if len(even) > 0:
        print(f"  Home win rate: {(even['FTR']=='H').mean()*100:.1f}% ({len(even)} matches)")
        print(f"  Draw rate: {(even['FTR']=='D').mean()*100:.1f}%")
else:
    print("⚠️ Shots data not available")

# ================================================
# TEST 3: ODDS MOVEMENT
# When closing odds differ significantly from opening
# ================================================
print("\n=== TEST 3: ODDS MOVEMENT ===")

# Check if we have both opening and closing odds
# Football-data.org has B365 (opening) and Avg/Max (closing proxy)
odds_move_df = df.dropna(subset=['B365H', 'AvgH', 'FTR']) if 'AvgH' in df.columns else pd.DataFrame()

if len(odds_move_df) > 0:
    odds_move_df = odds_move_df.copy()
    # Movement: closing shorter than opening = money coming in on home
    odds_move_df['HomeOddsMove'] = odds_move_df['B365H'] - odds_move_df['AvgH']

    # Significant movement toward home (odds shortened by 0.3+)
    home_money = odds_move_df[odds_move_df['HomeOddsMove'] > 0.3]
    against_home = odds_move_df[odds_move_df['HomeOddsMove'] < -0.3]

    if len(home_money) > 0:
        print(f"Money coming in on home team (odds shortened 0.3+):")
        print(f"  Home win rate: {(home_money['FTR']=='H').mean()*100:.1f}% ({len(home_money)} matches)")

    if len(against_home) > 0:
        print(f"Money going against home team (odds drifted 0.3+):")
        print(f"  Home win rate: {(against_home['FTR']=='H').mean()*100:.1f}% ({len(against_home)} matches)")

    if len(home_money) > 0 and len(against_home) > 0:
        diff = abs((home_money['FTR']=='H').mean() - (against_home['FTR']=='H').mean()) * 100
        print(f"  Difference: {diff:.1f} percentage points")
        if diff >= 3:
            print("✅ Odds movement IS worth adding to model")
        else:
            print("⚠️ Odds movement has minimal impact")
else:
    print("⚠️ Closing odds data not available for movement calculation")
    print("  B365 opening odds available but no closing odds column found")

# ================================================
# HONEST SUMMARY
# ================================================
print("\n=== HONEST SUMMARY ===")
print("Rule: Only add features with 3+ percentage point difference")
print("Anything below that adds noise not signal to the model")

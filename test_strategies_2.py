"""
BetWise Strategy Test 2
Tests three specific factors:
1. Referee tendencies (cards/fouls)
2. Basic odds value (upset potential)
3. Fixture congestion (days between matches)

HOW TO USE:
Run: python test_strategies_2.py
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
# TEST 1: REFEREE TENDENCIES
# Do high card referees favour home teams?
# ================================================
print("\n=== TEST 1: REFEREE TENDENCIES ===")

if 'HY' in df.columns and 'AY' in df.columns:
    df['TotalCards'] = df['HY'].fillna(0) + df['AY'].fillna(0) + \
                       df['HR'].fillna(0) + df['AR'].fillna(0)

    # Calculate referee average cards
    ref_avg = df.groupby('Referee')['TotalCards'].mean()
    df['RefAvgCards'] = df['Referee'].map(ref_avg)

    # High card referees vs low card referees
    median_cards = df['RefAvgCards'].median()

    high_card_ref = df[df['RefAvgCards'] > median_cards].dropna(subset=['FTR'])
    low_card_ref = df[df['RefAvgCards'] <= median_cards].dropna(subset=['FTR'])

    if len(high_card_ref) > 0 and len(low_card_ref) > 0:
        high_home = (high_card_ref['FTR'] == 'H').mean() * 100
        low_home = (low_card_ref['FTR'] == 'H').mean() * 100
        print(f"High card referees — Home win rate: {high_home:.1f}%")
        print(f"Low card referees  — Home win rate: {low_home:.1f}%")
        print(f"Difference: {abs(high_home - low_home):.1f} percentage points")

        if abs(high_home - low_home) >= 3:
            print("✅ Referee tendency HAS meaningful impact on home wins")
        else:
            print("⚠️ Referee tendency has minimal impact on home wins")
else:
    print("⚠️ Card data not available in dataset")

# ================================================
# TEST 2: ODDS VALUE / UPSET POTENTIAL
# When does the underdog actually win?
# ================================================
print("\n=== TEST 2: ODDS VALUE / UPSET POTENTIAL ===")

odds_df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTR'])

# Define underdog as team with odds > 3.0
home_underdog = odds_df[odds_df['B365H'] > 3.0]
away_underdog = odds_df[odds_df['B365A'] > 3.0]

if len(home_underdog) > 0:
    home_upset = (home_underdog['FTR'] == 'H').mean() * 100
    home_draw = (home_underdog['FTR'] == 'D').mean() * 100
    print(f"Home team underdog (odds > 3.0):")
    print(f"  Won: {home_upset:.1f}% | Drew: {home_draw:.1f}% | Won or Drew: {home_upset+home_draw:.1f}%")
    print(f"  Matches: {len(home_underdog)}")

if len(away_underdog) > 0:
    away_upset = (away_underdog['FTR'] == 'A').mean() * 100
    away_draw = (away_underdog['FTR'] == 'D').mean() * 100
    print(f"Away team underdog (odds > 3.0):")
    print(f"  Won: {away_upset:.1f}% | Drew: {away_draw:.1f}% | Won or Drew: {away_upset+away_draw:.1f}%")
    print(f"  Matches: {len(away_underdog)}")

# Big underdogs odds > 5.0
big_home_dog = odds_df[odds_df['B365H'] > 5.0]
big_away_dog = odds_df[odds_df['B365A'] > 5.0]

print(f"\nBig underdogs (odds > 5.0):")
if len(big_home_dog) > 0:
    print(f"  Home big underdog won: {(big_home_dog['FTR']=='H').mean()*100:.1f}% ({len(big_home_dog)} matches)")
if len(big_away_dog) > 0:
    print(f"  Away big underdog won: {(big_away_dog['FTR']=='A').mean()*100:.1f}% ({len(big_away_dog)} matches)")

# ================================================
# TEST 3: FIXTURE CONGESTION
# Do teams playing 3+ matches in 10 days perform worse?
# ================================================
print("\n=== TEST 3: FIXTURE CONGESTION ===")

# Calculate days since last match for each team
team_last_match = {}
home_days_rest = []
away_days_rest = []

for idx, row in df.iterrows():
    ht = row['HomeTeam']
    at = row['AwayTeam']
    current_date = row['Date']

    # Days since last match
    h_last = team_last_match.get(ht)
    a_last = team_last_match.get(at)

    h_days = (current_date - h_last).days if h_last else 99
    a_days = (current_date - a_last).days if a_last else 99

    home_days_rest.append(h_days)
    away_days_rest.append(a_days)

    # Update last match date
    team_last_match[ht] = current_date
    team_last_match[at] = current_date

df['HomeDaysRest'] = home_days_rest
df['AwayDaysRest'] = away_days_rest

# Congested = less than 4 days rest
home_congested = df[(df['HomeDaysRest'] < 4) & (df['HomeDaysRest'] > 0)]
home_rested = df[df['HomeDaysRest'] >= 7]

away_congested = df[(df['AwayDaysRest'] < 4) & (df['AwayDaysRest'] > 0)]
away_rested = df[df['AwayDaysRest'] >= 7]

print(f"Home team congested (< 4 days rest):")
if len(home_congested) > 0:
    hc_win = (home_congested['FTR'] == 'H').mean() * 100
    hr_win = (home_rested['FTR'] == 'H').mean() * 100
    print(f"  Win rate when congested: {hc_win:.1f}% ({len(home_congested)} matches)")
    print(f"  Win rate when rested (7+ days): {hr_win:.1f}% ({len(home_rested)} matches)")
    print(f"  Difference: {abs(hc_win - hr_win):.1f} percentage points")

print(f"\nAway team congested (< 4 days rest):")
if len(away_congested) > 0:
    ac_win = (away_congested['FTR'] == 'A').mean() * 100
    ar_win = (away_rested['FTR'] == 'A').mean() * 100
    print(f"  Win rate when congested: {ac_win:.1f}% ({len(away_congested)} matches)")
    print(f"  Win rate when rested (7+ days): {ar_win:.1f}% ({len(away_rested)} matches)")
    print(f"  Difference: {abs(ac_win - ar_win):.1f} percentage points")

# ================================================
# HONEST SUMMARY
# ================================================
print("\n=== HONEST SUMMARY ===")
print("Which factors are worth adding to the model?")
print("")
print("Referee tendencies: Check difference above")
print("Odds value/upsets:  Underdogs win less often than you think")
print("Fixture congestion: Check difference above")
print("")
print("Rule: Only add to model if difference is 3+ percentage points")
print("Smaller differences add noise not signal")

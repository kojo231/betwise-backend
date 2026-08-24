"""
BetWise Feature Engineering Script
Calculates prediction features for ALL leagues properly.
This ensures every league has the same quality of data
as the Premier League model.

HOW TO USE:
Run: python build_features.py

NOTE: This may take 10-20 minutes to process 64,000+ matches.
"""

import pandas as pd
import numpy as np

INPUT_FILE = "Master_BetWise_Final.csv"
OUTPUT_FILE = "Master_BetWise_Features.csv"

def build_features():
    print("=== BetWise Feature Builder ===\n")

    # Load dataset
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total matches: {len(df)}")
    print(f"Leagues: {df['League'].nunique()}")

    # Parse dates
    print("\nParsing dates...")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)

    # Initialize feature columns
    print("Initializing feature columns...")
    df['HomeForm'] = np.nan
    df['AwayForm'] = np.nan
    df['HomeGS'] = np.nan
    df['AwayGS'] = np.nan
    df['HomeGC'] = np.nan
    df['AwayGC'] = np.nan
    df['HomeStreak'] = np.nan
    df['AwayStreak'] = np.nan
    df['MomentumDiff'] = np.nan
    df['FormDiff'] = np.nan
    df['HomeAvgGoals'] = np.nan
    df['AwayAvgGoals'] = np.nan

    # Process each league separately
    leagues = df['League'].unique()
    print(f"\nProcessing {len(leagues)} leagues...\n")

    for league in leagues:
        print(f"Processing: {league}")

        # Get league matches
        league_idx = df[df['League'] == league].index.tolist()
        league_df = df.loc[league_idx].copy()

        home_history = {}
        away_history = {}
        team_streak = {}
        team_goals = {}

        processed = 0

        for idx in league_idx:
            row = df.loc[idx]
            ht = row['HomeTeam']
            at = row['AwayTeam']

            # Skip if no result
            if pd.isna(row['FTR']):
                continue

            # Home team stats (last 5 home games)
            h_hist = home_history.get(ht, [])[-5:]
            if h_hist:
                df.at[idx, 'HomeForm'] = sum(x[0] for x in h_hist)
                df.at[idx, 'HomeGS'] = sum(x[1] for x in h_hist)
                df.at[idx, 'HomeGC'] = sum(x[2] for x in h_hist)
                df.at[idx, 'HomeAvgGoals'] = sum(x[1] for x in h_hist) / len(h_hist)

            # Away team stats (last 5 away games)
            a_hist = away_history.get(at, [])[-5:]
            if a_hist:
                df.at[idx, 'AwayForm'] = sum(x[0] for x in a_hist)
                df.at[idx, 'AwayGS'] = sum(x[1] for x in a_hist)
                df.at[idx, 'AwayGC'] = sum(x[2] for x in a_hist)
                df.at[idx, 'AwayAvgGoals'] = sum(x[1] for x in a_hist) / len(a_hist)

            # Streaks
            h_streak = team_streak.get(ht, 0)
            a_streak = team_streak.get(at, 0)
            df.at[idx, 'HomeStreak'] = h_streak
            df.at[idx, 'AwayStreak'] = a_streak
            df.at[idx, 'MomentumDiff'] = h_streak - a_streak

            # Form difference
            h_form = df.at[idx, 'HomeForm']
            a_form = df.at[idx, 'AwayForm']
            if not pd.isna(h_form) and not pd.isna(a_form):
                df.at[idx, 'FormDiff'] = h_form - a_form

            # Update histories AFTER recording
            ftr = row['FTR']
            fthg = row['FTHG'] if not pd.isna(row.get('FTHG', np.nan)) else 0
            ftag = row['FTAG'] if not pd.isna(row.get('FTAG', np.nan)) else 0

            if ftr == 'H': h_pts, a_pts = 3, 0
            elif ftr == 'D': h_pts, a_pts = 1, 1
            else: h_pts, a_pts = 0, 3

            if ht not in home_history: home_history[ht] = []
            if at not in away_history: away_history[at] = []
            home_history[ht].append((h_pts, fthg, ftag))
            away_history[at].append((a_pts, ftag, fthg))

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

            processed += 1

        print(f"  ✅ {processed} matches processed")

    # Calculate bookmaker probabilities where odds exist
    print("\nCalculating bookmaker probabilities...")
    mask = df['B365H'].notna() & df['B365D'].notna() & df['B365A'].notna()
    df.loc[mask, 'ProbH'] = 1 / df.loc[mask, 'B365H']
    df.loc[mask, 'ProbD'] = 1 / df.loc[mask, 'B365D']
    df.loc[mask, 'ProbA'] = 1 / df.loc[mask, 'B365A']

    # Calculate expected goals
    mask2 = df['HomeAvgGoals'].notna() & df['AwayAvgGoals'].notna()
    df.loc[mask2, 'ExpectedGoals'] = df.loc[mask2, 'HomeAvgGoals'] + df.loc[mask2, 'AwayAvgGoals']

    # Format date back
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ FEATURE BUILDING COMPLETE!")
    print(f"Total matches: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\n📈 Features built per league:")
    for league in leagues:
        league_data = df[df['League'] == league]
        form_coverage = league_data['HomeForm'].notna().sum()
        total = len(league_data)
        pct = round(form_coverage/total*100, 1)
        print(f"  {league}: {form_coverage}/{total} matches with form data ({pct}%)")
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print("\nNext step: Update app.py to use Master_BetWise_Features.csv")

if __name__ == "__main__":
    build_features()

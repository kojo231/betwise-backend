"""
BetWise Deduplication Script
Removes duplicate matches from Master_All_Leagues.csv

HOW TO USE:
Run: python deduplicate.py
"""

import pandas as pd

INPUT_FILE = "Master_All_Leagues.csv"
OUTPUT_FILE = "Master_All_Leagues_Clean.csv"

def deduplicate():
    print("=== BetWise Deduplication ===\n")

    # Load dataset
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total matches before deduplication: {len(df)}")

    # Create a unique key for each match
    # A match is unique based on Date + HomeTeam + AwayTeam + League
    df['_key'] = (
        df['Date'].astype(str) + '_' +
        df['HomeTeam'].astype(str) + '_' +
        df['AwayTeam'].astype(str) + '_' +
        df['League'].astype(str)
    )

    # Count duplicates before removing
    duplicates = df.duplicated(subset=['_key'], keep='first').sum()
    print(f"Duplicate matches found: {duplicates}")

    # Remove duplicates keeping first occurrence
    df_clean = df.drop_duplicates(subset=['_key'], keep='first')

    # Drop the helper key column
    df_clean = df_clean.drop(columns=['_key'])

    # Sort by date
    try:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=True, format='mixed')
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        df_clean['Date'] = df_clean['Date'].dt.strftime('%d/%m/%Y')
    except Exception as e:
        print(f"Date sorting warning: {e}")

    # Save clean file
    df_clean.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ DEDUPLICATION COMPLETE!")
    print(f"Before: {len(df)} matches")
    print(f"Removed: {duplicates} duplicates")
    print(f"After: {len(df_clean)} matches")
    print(f"\n📈 Clean breakdown by league:")
    league_counts = df_clean.groupby('League').size().sort_values(ascending=False)
    for league, count in league_counts.items():
        print(f"  {league}: {count} matches")
    print(f"\n💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    deduplicate()

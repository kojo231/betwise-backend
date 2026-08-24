"""
BetWise Season Merge Script
Use this to add new season CSV files from Football-Data.co.uk
to your master dataset.

HOW TO USE:
1. Download latest season CSV from football-data.co.uk
2. Save it in the same folder as this script
3. Run: python merge_new_season.py
"""

import pandas as pd
import os
import glob

# Core columns we keep from each season file
CORE_COLUMNS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'Referee',
    'HS', 'AS',
    'HST', 'AST',
    'HF', 'AF',
    'HC', 'AC',
    'HY', 'AY',
    'HR', 'AR',
    'B365H', 'B365D', 'B365A',
]

MASTER_FILE = "Master_Premier_League_Final.csv"

def merge_new_file(new_file_path, season_label):
    """Add a new season CSV to the master dataset"""

    # Load master dataset
    if not os.path.exists(MASTER_FILE):
        print(f"Master file not found: {MASTER_FILE}")
        return

    master = pd.read_csv(MASTER_FILE)
    print(f"Master dataset: {len(master)} matches")

    # Load new file
    try:
        new_df = pd.read_csv(new_file_path, encoding='utf-8', low_memory=False)
    except:
        new_df = pd.read_csv(new_file_path, encoding='latin1', low_memory=False)

    print(f"New file: {len(new_df)} matches")

    # Keep only available core columns
    available = [col for col in CORE_COLUMNS if col in new_df.columns]
    new_df = new_df[available].copy()
    new_df.insert(0, 'Season', season_label)

    # Drop rows with no result
    new_df = new_df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])

    # Check for duplicates
    master_keys = set(
        master['Date'].astype(str) + '_' +
        master['HomeTeam'] + '_' +
        master['AwayTeam']
    )
    new_keys = (
        new_df['Date'].astype(str) + '_' +
        new_df['HomeTeam'] + '_' +
        new_df['AwayTeam']
    )
    truly_new = new_df[~new_keys.isin(master_keys)]

    if truly_new.empty:
        print("No new matches to add — all already in dataset")
        return

    print(f"Adding {len(truly_new)} new matches...")

    # Add missing columns with defaults
    for col in master.columns:
        if col not in truly_new.columns:
            truly_new = truly_new.copy()
            truly_new[col] = None

    # Combine and save
    updated = pd.concat([master, truly_new[master.columns]], ignore_index=True)
    updated = updated.sort_values('Date').reset_index(drop=True)

    # Backup old file first
    backup = MASTER_FILE.replace('.csv', '_backup.csv')
    master.to_csv(backup, index=False)
    print(f"Backup saved: {backup}")

    # Save updated file
    updated.to_csv(MASTER_FILE, index=False)

    print(f"\n✅ Dataset updated successfully!")
    print(f"Previous: {len(master)} matches")
    print(f"Added: {len(truly_new)} matches")
    print(f"Total now: {len(updated)} matches")


def auto_detect_and_merge(non_interactive_season: str | None = None):
    """Auto detect any new CSV files in the folder and merge them

    If non_interactive_season is provided, the script will not prompt for season labels.
    """


    print("=== BetWise Data Merger ===\n")

    # Find all CSV files except the master
    csv_files = [f for f in glob.glob("*.csv")
                 if f != MASTER_FILE
                 and 'backup' not in f.lower()
                 and 'column' not in f.lower()]

    if not csv_files:
        print("No new CSV files found in this folder.")
        print("Download a season CSV from football-data.co.uk")
        print("and save it in this folder, then run this script again.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process:")
    for f in csv_files:
        print(f"  - {f}")

    print()

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")

        # Determine season label
        if non_interactive_season is not None:
            season = non_interactive_season.strip() or "2025/26"
        else:
            season = input(f"Enter season label for {csv_file} (e.g. 2025/26): ").strip()
            if not season:
                season = "2025/26"


        merge_new_file(csv_file, season)
        print()

    print("Done! Now restart your backend with: python app.py")


if __name__ == "__main__":
    # Usage:
    #   python merge_new_season.py                      (interactive)
    #   python merge_new_season.py --non-interactive   (uses 2025/26 for all season labels)
    import sys
    non_interactive = None
    if "--non-interactive" in sys.argv:
        non_interactive = "2025/26"


    auto_detect_and_merge(non_interactive_season=non_interactive)


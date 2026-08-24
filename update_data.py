"""
BetWise Data Updater
Fetches recent Premier League results and updates the dataset automatically.
Run this script regularly to keep predictions accurate.
"""

import requests
import pandas as pd
from datetime import datetime, date
import os

API_KEY = "123"
API_BASE = "https://www.thesportsdb.com/api/v1/json"

# Team name mapping - TheSportsDB to our dataset format
TEAM_MAP = {
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Nottingham Forest": "Nottm Forest",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Sheffield United": "Sheffield Utd",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Queens Park Rangers": "QPR",
    "Cardiff City": "Cardiff",
}

def map_team(name):
    return TEAM_MAP.get(name, name)

def fetch_recent_results():
    """Fetch recent completed Premier League matches"""
    print("Fetching recent results from TheSportsDB...")
    try:
        url = f"{API_BASE}/{API_KEY}/eventspastleague.php?id=4328"
        res = requests.get(url, timeout=10)
        data = res.json()
        events = data.get("events", []) or []
        print(f"Found {len(events)} recent matches")
        return events
    except Exception as e:
        print(f"Error fetching results: {e}")
        return []

def process_events(events):
    """Convert TheSportsDB events to our dataset format"""
    rows = []
    for m in events:
        # Only include completed matches
        if m.get("intHomeScore") is None or m.get("intAwayScore") is None:
            continue
        if m.get("strStatus") not in ["Match Finished", "FT"]:
            continue

        home_score = int(m["intHomeScore"])
        away_score = int(m["intAwayScore"])

        if home_score > away_score: ftr = "H"
        elif away_score > home_score: ftr = "A"
        else: ftr = "D"

        row = {
            "Season": "2025/26",
            "Date": m.get("dateEvent", ""),
            "HomeTeam": map_team(m.get("strHomeTeam", "")),
            "AwayTeam": map_team(m.get("strAwayTeam", "")),
            "FTHG": home_score,
            "FTAG": away_score,
            "FTR": ftr,
            "HTHG": 0,  # Not available in free tier
            "HTAG": 0,
            "HTR": "D",
            "B365H": 2.0,  # Default odds if not available
            "B365D": 3.4,
            "B365A": 4.0,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def update_dataset():
    """Main update function"""
    dataset_path = "Master_Premier_League_Final.csv"

    # Load existing dataset
    if os.path.exists(dataset_path):
        existing = pd.read_csv(dataset_path)
        print(f"Existing dataset: {len(existing)} matches")
    else:
        print("No existing dataset found!")
        return

    # Fetch new results
    events = fetch_recent_results()
    if not events:
        print("No new results to add")
        return

    # Process new results
    new_data = process_events(events)
    if new_data.empty:
        print("No completed matches found")
        return

    print(f"Processing {len(new_data)} new matches...")

    # Check which matches are already in dataset
    existing_keys = set(
        existing["Date"].astype(str) + "_" +
        existing["HomeTeam"] + "_" +
        existing["AwayTeam"]
    )

    new_keys = (
        new_data["Date"].astype(str) + "_" +
        new_data["HomeTeam"] + "_" +
        new_data["AwayTeam"]
    )

    truly_new = new_data[~new_keys.isin(existing_keys)]

    if truly_new.empty:
        print("All matches already in dataset — no update needed")
        return

    print(f"Adding {len(truly_new)} new matches to dataset...")

    # Add missing columns with defaults
    for col in existing.columns:
        if col not in truly_new.columns:
            truly_new[col] = None

    # Combine and save
    updated = pd.concat([existing, truly_new[existing.columns]], ignore_index=True)
    updated = updated.sort_values("Date").reset_index(drop=True)
    updated.to_csv(dataset_path, index=False)

    print(f"\n✅ Dataset updated successfully!")
    print(f"Previous matches: {len(existing)}")
    print(f"New matches added: {len(truly_new)}")
    print(f"Total matches now: {len(updated)}")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show new matches added
    print("\nNew matches added:")
    for _, row in truly_new.iterrows():
        print(f"  {row['Date']} | {row['HomeTeam']} {row['FTHG']}-{row['FTAG']} {row['AwayTeam']} | {row['FTR']}")

if __name__ == "__main__":
    update_dataset()

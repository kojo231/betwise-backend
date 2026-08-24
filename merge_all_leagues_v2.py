"""
BetWise Updated Merge Script
Same as merge_all_leagues.py but includes closing odds columns.

HOW TO USE:
1. Put all league CSV files in 'all_leagues' folder
2. Run: python merge_all_leagues_v2.py
"""

import pandas as pd
import glob
import os

LEAGUE_MAP = {
    'E0': 'England Premier League',
    'SP1': 'Spain La Liga',
    'D1': 'Germany Bundesliga',
    'I1': 'Italy Serie A',
    'F1': 'France Ligue 1',
    'P1': 'Portugal Primeira Liga',
    'N1': 'Netherlands Eredivisie',
    'B1': 'Belgium Jupiler League',
    'T1': 'Turkey Ligi 1',
    'G1': 'Greece Ethniki',
    'SC0': 'Scotland Premiership',
}

# Updated standard columns including closing odds
STANDARD_COLUMNS = [
    'League', 'Season', 'Div',
    'Date', 'Time',
    'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR',
    # Opening odds
    'B365H', 'B365D', 'B365A',
    'AvgH', 'AvgD', 'AvgA',
    'MaxH', 'MaxD', 'MaxA',
    # Closing odds — key addition
    'B365CH', 'B365CD', 'B365CA',
    'AvgCH', 'AvgCD', 'AvgCA',
    'MaxCH', 'MaxCD', 'MaxCA',
    # Over/Under
    'B365>2.5', 'B365<2.5',
    'Avg>2.5', 'Avg<2.5',
    'B365C>2.5', 'B365C<2.5',
    'AvgC>2.5', 'AvgC<2.5',
    # Asian handicap
    'B365AHH', 'B365AHA',
    'AvgAHH', 'AvgAHA',
]

def detect_season(filename, df):
    name = filename.lower()
    seasons = [
        '2025-26', '2024-25', '2023-24', '2022-23', '2021-22',
        '2020-21', '2019-20', '2018-19', '2017-18', '2016-17',
        '2015-16', '2014-15', '2013-14', '2012-13', '2011-12',
        '2010-11', '2009-10', '2008-09', '2007-08', '2006-07',
        '2005-06', '2004-05', '2003-04', '2002-03', '2001-02',
        '2000-01', '1999-00', '1998-99', '1997-98', '1996-97',
        '1995-96', '1994-95', '1993-94',
    ]
    for s in seasons:
        if s in name:
            parts = s.split('-')
            return f"20{parts[0][-2:]}/{'20' if int(parts[1]) < 50 else '19'}{parts[1]}"
    if 'Date' in df.columns and len(df) > 0:
        try:
            date = pd.to_datetime(df['Date'].dropna().iloc[0], dayfirst=True)
            year = date.year
            month = date.month
            if month >= 7:
                return f"{year}/{str(year+1)[-2:]}"
            else:
                return f"{year-1}/{str(year)[-2:]}"
        except:
            pass
    return "Unknown"

def detect_league(div_code):
    if pd.isna(div_code):
        return "Unknown"
    return LEAGUE_MAP.get(str(div_code).strip(), str(div_code))

def merge_all_leagues():
    print("=== BetWise Multi-League Merger v2 (with closing odds) ===\n")

    folder = "all_leagues"
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found!")
        return

    csv_files = glob.glob(f"{folder}/*.csv")
    if not csv_files:
        print(f"No CSV files found in '{folder}' folder!")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    all_data = []

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)

        try:
            try:
                df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            except:
                df = pd.read_csv(filepath, encoding='latin1', low_memory=False)

            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])

            if len(df) == 0:
                print(f"  ⚠️ {filename} — no valid matches, skipping")
                continue

            div_code = df['Div'].iloc[0] if 'Div' in df.columns else 'Unknown'
            league_name = detect_league(div_code)
            season = detect_season(filename, df)

            # Check for closing odds
            has_closing = 'B365CH' in df.columns
            closing_str = "✅ closing odds" if has_closing else "❌ no closing odds"

            # Build output with standard columns
            output = pd.DataFrame(index=range(len(df)), columns=STANDARD_COLUMNS)
            output['League'] = league_name
            output['Season'] = season

            for col in STANDARD_COLUMNS:
                if col in df.columns and col not in ['League', 'Season']:
                    output[col] = df[col].values

            print(f"  ✅ {league_name} — {season} — {len(df)} matches — {closing_str}")
            all_data.append(output)

        except Exception as e:
            print(f"  ❌ Error: {filename}: {e}")
            continue

    if not all_data:
        print("\nNo data to merge!")
        return

    print(f"\nMerging {len(all_data)} files...")
    master = pd.concat(all_data, ignore_index=True)

    # Remove duplicates
    master['_key'] = (master['Date'].astype(str) + '_' +
                      master['HomeTeam'].astype(str) + '_' +
                      master['AwayTeam'].astype(str) + '_' +
                      master['League'].astype(str))
    before = len(master)
    master = master.drop_duplicates(subset=['_key'], keep='first')
    master = master.drop(columns=['_key'])
    removed = before - len(master)

    # Sort by date
    try:
        master['Date'] = pd.to_datetime(master['Date'], dayfirst=True, format='mixed')
        master = master.sort_values('Date').reset_index(drop=True)
        master['Date'] = master['Date'].dt.strftime('%d/%m/%Y')
    except Exception as e:
        print(f"Date sorting warning: {e}")

    output_path = "Master_All_Leagues_v2.csv"
    master.to_csv(output_path, index=False)

    print(f"\n✅ MERGE COMPLETE!")
    print(f"📊 Total matches: {len(master)}")
    print(f"🗑️ Duplicates removed: {removed}")
    print(f"📋 Total columns: {len(master.columns)}")

    # Show closing odds coverage
    if 'B365CH' in master.columns:
        closing_coverage = master['B365CH'].notna().sum()
        print(f"📈 Matches with closing odds: {closing_coverage} ({closing_coverage/len(master)*100:.1f}%)")

    print(f"\n📈 Breakdown by league:")
    league_counts = master.groupby('League').size().sort_values(ascending=False)
    for league, count in league_counts.items():
        print(f"  {league}: {count} matches")

    print(f"\n💾 Saved to: {output_path}")
    print("\nNext step: Run build_features.py on the new file")

if __name__ == "__main__":
    merge_all_leagues()

"""
BetWise Multi-League Merge Script
Merges all league CSV files into one clean master dataset.

HOW TO USE:
1. Put all your league CSV files in a folder called "all_leagues"
   inside your backend folder
2. Run: python merge_all_leagues.py
"""

import pandas as pd
import glob
import os

# League name mapping based on Div code
LEAGUE_MAP = {
    'E0': 'England Premier League',
    'E1': 'England Championship',
    'SP1': 'Spain La Liga',
    'SP2': 'Spain Segunda',
    'D1': 'Germany Bundesliga',
    'D2': 'Germany 2. Bundesliga',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'F1': 'France Ligue 1',
    'F2': 'France Ligue 2',
    'P1': 'Portugal Primeira Liga',
    'N1': 'Netherlands Eredivisie',
    'B1': 'Belgium Jupiler League',
    'T1': 'Turkey Ligi 1',
    'G1': 'Greece Ethniki',
    'SC0': 'Scotland Premiership',
    'A1': 'Austria Bundesliga',
    'RU1': 'Russia Premier League',
    'SW1': 'Switzerland Super League',
    'N1': 'Norway Eliteserien',
    'PL1': 'Poland Ekstraklasa',
    'R1': 'Romania Liga 1',
    'DK1': 'Denmark Superliga',
    'IR1': 'Ireland Premier Division',
    'SE1': 'Sweden Allsvenskan',
}

# Core columns that matter for prediction
# Using newer season standard — older seasons will have empty fields
STANDARD_COLUMNS = [
    'League', 'Season', 'Div',
    'Date', 'Time',
    'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR',
    'B365H', 'B365D', 'B365A',
    'MaxH', 'MaxD', 'MaxA',
    'AvgH', 'AvgD', 'AvgA',
    'B365>2.5', 'B365<2.5',
    'Avg>2.5', 'Avg<2.5',
    'AHh', 'B365AHH', 'B365AHA',
]

def detect_season(filename, df):
    """Try to detect season from filename or data"""
    name = filename.lower()
    
    # Common season patterns in filenames
    seasons = [
        '2024-25', '2023-24', '2022-23', '2021-22',
        '2020-21', '2019-20', '2018-19', '2017-18',
        '2016-17', '2015-16', '2014-15', '2013-14',
        '2012-13', '2011-12', '2010-11', '2009-10',
    ]
    
    for s in seasons:
        if s in name:
            parts = s.split('-')
            return f"20{parts[0][-2:]}/{'20' if int(parts[1]) < 50 else '19'}{parts[1]}"
    
    # Try to detect from date column
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
    """Get league name from div code"""
    if pd.isna(div_code):
        return "Unknown"
    return LEAGUE_MAP.get(str(div_code).strip(), str(div_code))

def merge_all_leagues():
    print("=== BetWise Multi-League Merger ===\n")
    
    # Find all CSV files in all_leagues folder
    folder = "all_leagues"
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found!")
        print("Please create an 'all_leagues' folder and put your CSV files in it.")
        return
    
    csv_files = glob.glob(f"{folder}/*.csv")
    
    if not csv_files:
        print(f"No CSV files found in '{folder}' folder!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:\n")
    
    all_data = []
    
    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        # Read file
        try:
            try:
                df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            except:
                df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
            # Drop completely empty rows
            df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])
            
            if len(df) == 0:
                print(f"  ⚠️ No valid matches found — skipping")
                continue
            
            # Detect league and season
            div_code = df['Div'].iloc[0] if 'Div' in df.columns else 'Unknown'
            league_name = detect_league(div_code)
            season = detect_season(filename, df)
            
            # Add league and season columns
            df.insert(0, 'League', league_name)
            df.insert(1, 'Season', season)
            
            # Keep only standard columns that exist in this file
            # Columns not in this file will be added as empty (NaN)
            available = [col for col in STANDARD_COLUMNS if col in df.columns]
            missing = [col for col in STANDARD_COLUMNS if col not in df.columns]
            
            # Create output dataframe with all standard columns
            output = pd.DataFrame(columns=STANDARD_COLUMNS)
            for col in available:
                output[col] = df[col].values[:len(df)]
            
            # Missing columns stay as NaN — no zeros, no estimates
            
            print(f"  ✅ {league_name} — {season} — {len(df)} matches")
            if missing:
                print(f"  📭 Missing columns (left empty): {len(missing)}")
            
            all_data.append(output)
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    if not all_data:
        print("\nNo data to merge!")
        return
    
    # Merge all data
    print(f"\nMerging {len(all_data)} files...")
    master = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    try:
        master['Date'] = pd.to_datetime(master['Date'], dayfirst=True, format='mixed')
        master = master.sort_values('Date').reset_index(drop=True)
        master['Date'] = master['Date'].dt.strftime('%d/%m/%Y')
    except Exception as e:
        print(f"Date sorting warning: {e}")
    
    # Save
    output_path = "Master_All_Leagues.csv"
    master.to_csv(output_path, index=False)
    
    print(f"\n✅ MERGE COMPLETE!")
    print(f"📊 Total matches: {len(master)}")
    print(f"📋 Total columns: {len(master.columns)}")
    print(f"\n📈 Breakdown by league:")
    league_counts = master.groupby('League').size().sort_values(ascending=False)
    for league, count in league_counts.items():
        print(f"  {league}: {count} matches")
    print(f"\n💾 Saved to: {output_path}")
    print("\nNext step: Run python app.py to retrain the model on new data!")

if __name__ == "__main__":
    merge_all_leagues()

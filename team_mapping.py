"""
BetWise Team Name Mapping
Maps API team names to dataset team names.

Add this to your app.py to fix prediction accuracy
for teams with mismatched names.
"""

# Maps external API names to our dataset names
TEAM_MAP = {
    # Premier League
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Nottingham Forest": "Nottm Forest",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
    "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Wigan Athletic": "Wigan",
    "Sunderland AFC": "Sunderland",
    "Coventry City": "Coventry",
    "Middlesbrough FC": "Middlesbrough",

    # La Liga
    "Atletico Madrid": "Atletico Madrid",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Deportivo La Coruna": "La Coruna",
    "Rayo Vallecano": "Vallecano",
    "UD Almeria": "Almeria",
    "Girona FC": "Girona",
    "Las Palmas": "Las Palmas",

    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Borussia Monchengladbach": "M'gladbach",
    "Werder Bremen": "Werder Bremen",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "FC Augsburg": "Augsburg",
    "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Schalke 04": "Schalke",
    "Hamburger SV": "Hamburg",
    "Hertha Berlin": "Hertha",
    "FC Cologne": "Cologne",
    "FC Koln": "Cologne",
    "Fortuna Dusseldorf": "Dusseldorf",
    "SV Darmstadt 98": "Darmstadt",
    "Union Berlin": "Union Berlin",

    # Serie A
    "AC Milan": "AC Milan",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "Juventus FC": "Juventus",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Atalanta BC": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Bologna FC": "Bologna",
    "Cagliari Calcio": "Cagliari",
    "Genoa CFC": "Genoa",
    "Hellas Verona": "Verona",
    "Empoli FC": "Empoli",
    "US Sassuolo": "Sassuolo",
    "US Salernitana": "Salernitana",
    "Frosinone Calcio": "Frosinone",
    "Monza": "Monza",
    "Venezia FC": "Venezia",
    "Como 1907": "Como",

    # Ligue 1
    "Paris Saint-Germain": "PSG",
    "Paris SG": "PSG",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Stade Rennais": "Rennes",
    "RC Lens": "Lens",
    "Stade de Reims": "Reims",
    "RC Strasbourg": "Strasbourg",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "Girondins Bordeaux": "Bordeaux",
    "Toulouse FC": "Toulouse",
    "FC Metz": "Metz",
    "Clermont Foot": "Clermont",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "Stade Brestois": "Brest",
    "AS Saint-Etienne": "St Etienne",

    # Portugal
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting",
    "SC Braga": "Braga",
    "Vitoria SC": "Vitoria",
    "Boavista FC": "Boavista",
    "FC Famalicao": "Famalicao",
    "Moreirense FC": "Moreirense",
    "Rio Ave FC": "Rio Ave",
    "Gil Vicente FC": "Gil Vicente",
}

def map_team(name):
    """Map API team name to dataset team name"""
    if not name:
        return name
    # Direct match
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    # Try partial match
    for api_name, dataset_name in TEAM_MAP.items():
        if api_name.lower() in name.lower() or name.lower() in api_name.lower():
            return dataset_name
    # Return original if no match found
    return name

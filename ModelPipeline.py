#!/usr/bin/env python3
import torch
from model import PlayerImpactMatchupModel
import pandas as pd
from utils import  predict_matchup,convert_int_season_to_str,teams
import argparse
import sys
import os 
from dotenv import load_dotenv
load_dotenv()

# NBA Teams Mapping
teams = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}
MODEL_SAVE_PATH = os.getenv("MODELS_FOLDER", "./models")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./models")

parser = argparse.ArgumentParser(description="Predict NBA matchup win probabilities.")
parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 2017)")
parser.add_argument("--home", action="store_true", help="Set if the first team is at home")
parser.add_argument(
    "--matchup",
    nargs=2,
    required=True,
    help="Two team abbreviations (e.g., PHX TOR). Teams: " +
         ", ".join([f"{abbr} ({name})" for abbr, name in teams.items()])
)
args = parser.parse_args()

SEASON = args.season
IS_HOME = int(args.home)
MATCHUP = tuple(args.matchup)


# Recreate model with same params used in training:
player_feat_dim = 1  # e.g., 'playerImpact'
num_players = 17    # this value may vary based on the season of the trained model
context_dim = 3
hidden_dim = 64

model = PlayerImpactMatchupModel(
    num_players=num_players,
    context_dim=context_dim,
    hidden_dim=hidden_dim,
    dropout=0.3
)

model.load_state_dict(torch.load(MODEL_SAVE_PATH+"game_prediction_model.pth"))
model.eval()

# Then run your prediction function or inference here...
all_elos = pd.read_csv(DATA_FOLDER + "/gamesAndEloStats.csv")
playersStats = pd.read_csv(DATA_FOLDER + "/playerStats.csv")
player_count = min([
    len(playersStats[(playersStats['teamTricode'] == t) & 
                     (playersStats['season_year'] == convert_int_season_to_str(SEASON))])
    for t in teams
])

try:
    # Forward prediction
    t1P_fwd, t2P_fwd = predict_matchup(
        model=model,
        playersStats=playersStats,
        all_elos=all_elos,
        matchup=MATCHUP,
        season=SEASON,
        is_home=IS_HOME,
        player_count=player_count
    )

    # Reverse prediction
    t1P_rev, t2P_rev = predict_matchup(
        model=model,
        playersStats=playersStats,
        all_elos=all_elos,
        matchup=(MATCHUP[1], MATCHUP[0]),
        season=SEASON,
        is_home=1 - IS_HOME,
        player_count=player_count
    )

    # Average predictions
    team1_prob = torch.sigmoid(torch.tensor((t1P_fwd + t2P_rev) / 2))
    team2_prob = torch.sigmoid(torch.tensor((t2P_fwd + t1P_rev) / 2))

    print(f"\n📊 Matchup: {MATCHUP[0]} vs {MATCHUP[1]}")
    print(f"📅 Season: {convert_int_season_to_str(SEASON)}")
    print(f"🏠 Home Team: {MATCHUP[0 if IS_HOME else 1]}")
    print(f"🔄 Bidirectional Averaged Prediction")
    print(f"🔮 Win Probabilities:")
    print(f"   • {MATCHUP[0]}: {team1_prob:.3f}")
    print(f"   • {MATCHUP[1]}: {team2_prob:.3f}")

except Exception as e:
    print(f"❌ Error during prediction: {e}", file=sys.stderr)
    sys.exit(1)

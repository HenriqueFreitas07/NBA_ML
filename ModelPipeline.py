import torch
from model import FullPlayerTeamMatchupModel
import pandas as pd
from utils import  predict_matchup,convert_int_season_to_str,teams
DATA_FOLDER="./datasets/DATA_AGGREGATIONS/"
SEASON=2023
MATCHUP=("WAS","BOS")

# Recreate model with same params used in training:
player_feat_dim = 1  # e.g., 'playerImpact'
num_players = 17    # must match training
context_dim = 3
hidden_dim = 64

model = FullPlayerTeamMatchupModel(player_feat_dim, num_players, context_dim, hidden_dim)
model.load_state_dict(torch.load("nba_matchup_model.pth"))
model.eval()

# Then run your prediction function or inference here...
all_elos = pd.read_csv(DATA_FOLDER + "gamesAndEloStats.csv")
playersStats = pd.read_csv(DATA_FOLDER + "/playerStats.csv")
player_count = min([
    len(playersStats[(playersStats['teamTricode'] == t) & 
                     (playersStats['season_year'] == convert_int_season_to_str(SEASON))])
    for t in teams
])

prediction = predict_matchup(
    model=model, 
    playersStats=playersStats, 
    all_elos=all_elos,
    matchup=MATCHUP, 
    season=SEASON, 
    is_home=1,
    player_count=player_count
)

print(f"🏀 Win probability for {MATCHUP[0]} vs {MATCHUP[1]}: {prediction:.3f}")

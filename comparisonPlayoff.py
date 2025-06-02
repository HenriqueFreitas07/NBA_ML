import itertools
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from model import PlayerImpactMatchupModel
from utils import  predict_matchup,convert_int_season_to_str,teams,getMatchupByTeamBySeason
import os 
import numpy as np
import torch    
from dotenv import load_dotenv
import os
load_dotenv()

MODEL_SAVE_PATH = os.getenv("MODELS_FOLDER", "./models/")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./datasets/DATA_AGGREGATIONS")

player_feat_dim = 1  # e.g., 'playerImpact'
context_dim = 3
hidden_dim = 64

df = pd.read_csv("./datasets/NBA_DATA_2010_2024/play_off_totals_2010_2024.csv")
all_matchups = list(itertools.permutations(teams, 2))
data = dict()
SEASON=2021
all_elos = pd.read_csv(DATA_FOLDER + "/gamesAndEloStats.csv")
playersStats = pd.read_csv(DATA_FOLDER + "/playerStats.csv")
player_count = min([
    len(playersStats[(playersStats['teamTricode'] == t) & 
                    (playersStats['season_year'] == convert_int_season_to_str(SEASON))])
    for t in teams
])

model = PlayerImpactMatchupModel(
    num_players=player_count,
    context_dim=context_dim,
    hidden_dim=hidden_dim,
    dropout=0.3
)

model.load_state_dict(torch.load(MODEL_SAVE_PATH+"game_prediction_model.pth"))
model.eval()
matchupAccuracy=dict()
for m in all_matchups:
    values=getMatchupByTeamBySeason(df,m,season=SEASON)
    t1=m[0]
    t2=m[1]
    if values.empty:
        continue
    values["IS_HOME"] = values["IS_HOME"].astype(bool)
    home_t1 = values[(values["IS_HOME"]) & (values['TEAM_ABBREVIATION'] == t1)]
    away_t2 = values[(~values["IS_HOME"]) & (values['TEAM_ABBREVIATION'] == t2)]
    true_label_home=home_t1["WL"].to_list()
    true_label_away=away_t2["WL"].to_list()
    accuracy=0
    try:
        # Forward prediction
        t1P_fwd, t2P_fwd = predict_matchup(
            model=model,
            playersStats=playersStats,
            all_elos=all_elos,
            matchup=m,
            season=SEASON,
            is_home=1,
            player_count=player_count
            )
        team1_prob= torch.sigmoid(torch.tensor(t1P_fwd)) # win_pct home 
        team2_prob = torch.sigmoid(torch.tensor(t2P_fwd)) # win_pct away
        pred_classes =  1 if team1_prob > team2_prob else 0
        for current_game in true_label_home:
           accuracy+=int(current_game==pred_classes)
    except Exception as e :
        pass
    matchupAccuracy[m]=(accuracy/len(true_label_home))*100

media=sum([v for v in matchupAccuracy.values()])/len(matchupAccuracy)
display(matchupAccuracy)
display(media)
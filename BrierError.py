from utils import teams,get_head_to_head_win_pct
import itertools
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from model import PlayerImpactMatchupModel
from utils import  predict_matchup,convert_int_season_to_str,teams
import os 
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
SEASON=2022

brier_scores = []
matchup_labels = []

for m in all_matchups:
    values=get_head_to_head_win_pct(df,m,season=SEASON)
    if values[m[0]+"_home_win_pct"] is None:
        continue
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

        t1_win_pct_home=values[m[0]+"_home_win_pct"]
        t2_win_pct_away=values[m[1]+"_away_win_pct"]

        team1_prob = torch.sigmoid(torch.tensor((t1P_fwd)))
        team2_prob = torch.sigmoid(torch.tensor(t2P_fwd))
        # apply the brier score to the predictions and the values 
        brier_t1_fwd = (team1_prob.item() - t1_win_pct_home / 100) ** 2
        brier_t2_fwd = (team2_prob.item() - t2_win_pct_away / 100) ** 2
        # Reverse prediction
        t1P_rev, t2P_rev = predict_matchup(
            model=model,
            playersStats=playersStats,
            all_elos=all_elos,
            matchup=(m[1], m[0]),
            season=SEASON,
            is_home=0,
            player_count=player_count
        )

        t1_win_pct_away=values[m[0]+"_away_win_pct"]
        t2_win_pct_home=values[m[1]+"_home_win_pct"]

        team1_prob = torch.sigmoid(torch.tensor((t1P_rev)))
        team2_prob = torch.sigmoid(torch.tensor(t2P_rev))
        # apply the brier score to the predictions and the values 
        brier_t1_rev = (team1_prob.item() - t1_win_pct_away / 100) ** 2
        brier_t2_rev = (team2_prob.item() - t2_win_pct_home / 100) ** 2

        avg_brier = (brier_t1_fwd + brier_t2_fwd + brier_t1_rev + brier_t2_rev) / 4
        brier_scores.append(avg_brier)
        matchup_labels.append(f"{m[0]} vs {m[1]}")
    except Exception as e :
        pass

errors=list(zip(matchup_labels,brier_scores))
print(errors)

# Sort the errors by Brier score
sorted_errors = sorted(zip(matchup_labels, brier_scores), key=lambda x: x[1])
labels, scores = zip(*sorted_errors)

# Plot
plt.figure(figsize=(16, 6))
plt.bar(labels, scores)
plt.xticks(rotation=90)
plt.ylabel("Brier Score")
plt.title(f"Brier Scores por Matchup (Season {SEASON})")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
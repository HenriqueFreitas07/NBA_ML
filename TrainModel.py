from dotenv import load_dotenv
import os
load_dotenv()

MODEL_SAVE_PATH = os.getenv("MODELS_FOLDER", "./models/")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./datasets/DATA_AGGREGATIONS")

import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import PlayerImpactMatchupModel
from utils import (
    get_head_to_head_win_pct,
    convert_int_season_to_str,
    teams
)

# ---------------- PARAMETERS ---------------------
SEASON = 2021
player_features = ['playerImpact']  # only use impact
context_dim = 3  # [is_home, elo1, elo2]
num_epochs = 10
learning_rate = 1e-3

# ---------------- LOAD DATA ---------------------
regular_games_total = pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_totals_2010_2024.csv")
playersStats = pd.read_csv(DATA_FOLDER + "/playerStats.csv")
all_elos = pd.read_csv(DATA_FOLDER + "/gamesAndEloStats.csv")

# Get valid player count per team
player_count = min([
    len(playersStats[(playersStats['teamTricode'] == t) & 
                     (playersStats['season_year'] == convert_int_season_to_str(SEASON))])
    for t in teams
])

# ---------------- MODEL INIT ---------------------
model = PlayerImpactMatchupModel(num_players=player_count, context_dim=context_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# ---------------- TRAIN LOOP ---------------------
model.train()
all_matchups = list(itertools.permutations(teams, 2))

for epoch in range(num_epochs):
    total_loss = 0
    n_games = 0
    for team1, team2 in all_matchups:
        try:
            # PLAYER IMPACT VECTORS
            def get_impact(team):
                df = playersStats[(playersStats['teamTricode'] == team) & (playersStats['season_year'] == convert_int_season_to_str(SEASON))]
                vals = df[player_features].to_numpy(dtype=np.float32)[:player_count]
                if len(vals) < player_count:
                    raise ValueError("not enough players")
                return torch.tensor(vals).unsqueeze(0)  # [1, N, F], F=1 playerImpact

            p1_tensor = get_impact(team1)
            p2_tensor = get_impact(team2)

            # CONTEXT: is_home, elo1, elo2
            elos_df = all_elos[(all_elos['SEASON_YEAR'] == convert_int_season_to_str(SEASON)) & (all_elos['TEAM_ABBREVIATION'].isin([team1, team2]))]
            if len(elos_df) < 2:
                continue
            elo1 = float(elos_df[elos_df['TEAM_ABBREVIATION'] == team1]['elo_before_game'].iloc[0])
            elo2 = float(elos_df[elos_df['TEAM_ABBREVIATION'] == team2]['elo_before_game'].iloc[0])
            elo1 = (elo1 - 1300) / 500  # normalize
            elo2 = (elo2 - 1300) / 500
            is_home = int(f"{team2} @ {team1}" in regular_games_total['MATCHUP'].values)
            ctx_tensor = torch.tensor([[is_home, elo1, elo2]], dtype=torch.float32)

            # LABEL: P(team1 wins), P(team2 wins) smoothed from h2h
            h2h = get_head_to_head_win_pct(regular_games_total, (team1, team2), season=SEASON)
            p1_win = h2h.get(f"{team1 if is_home else team2}_home_win_pct")
            p2_win = h2h.get(f"{team1 if not is_home else team2}_away_win_pct")
            if p1_win is None or p2_win is None:
                continue
            label = torch.tensor([[p1_win, p2_win]], dtype=torch.float32)

            # TRAIN STEP
            pred = model(p1_tensor, p2_tensor, ctx_tensor)
            loss = loss_fn(pred, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            n_games += 1
        except Exception as e:
            print(f"Skip ({team1}, {team2}) → {e}")

    print(f"Epoch {epoch+1}: Loss = {total_loss / max(n_games,1):.4f}  on {n_games} matchups")

# ---------------- SAVE MODEL ---------------------
path=MODEL_SAVE_PATH+"/game_prediction_model.pth"
torch.save(model.state_dict(),path)
print(f"Model saved → {path}")
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import FullPlayerTeamMatchupModel  # import the model
from utils import (
    get_head_to_head_win_pct,
    convert_int_season_to_str,
    teams
)

# ---------------- PARAMETERS ---------------------
DATA_FOLDER = "./datasets/DATA_AGGREGATIONS/"
SEASON = 2022 
IS_HOME = 1
player_features = ['playerImpact']
num_epochs = 7
learning_rate = 0.001
hidden_dim = 64

# Load data

all_possible_matchups = list(itertools.combinations(teams, 2))

regular_games_total = pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_totals_2010_2024.csv")
regular_season_all_parts = pd.concat([
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_1.csv"),
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_2.csv"),
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_3.csv")
])
all_elos = pd.read_csv(DATA_FOLDER + "gamesAndEloStats.csv")
playersStats = pd.read_csv(DATA_FOLDER + "/playerStats.csv")

# Get fixed player count for the model
player_count = min([
    len(playersStats[(playersStats['teamTricode'] == t) & 
                     (playersStats['season_year'] == convert_int_season_to_str(SEASON))])
    for t in teams
])

player_feat_dim = len(player_features)
context_dim = 3  # [homeAway, elo_1, elo_2]

# Initialize model, optimizer, loss
model = FullPlayerTeamMatchupModel(player_feat_dim, player_count, context_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# ---------------- TRAINING LOOP ---------------------
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    num_games = 0
    for matchup in all_possible_matchups:
        try:
            # Extract player stats
            team_stats = []
            for t in matchup:
                team_df = playersStats[(playersStats['teamTricode'] == t) & 
                                       (playersStats['season_year'] == convert_int_season_to_str(SEASON))]
                if len(team_df) < player_count:
                    raise ValueError(f"Not enough players for {t}")
                stats = team_df[player_features].to_numpy(dtype=np.float32)[:player_count]
                team_stats.append(stats)
            
            team1_tensor = torch.tensor(team_stats[0]).unsqueeze(0)  # [1, N, F]
            team2_tensor = torch.tensor(team_stats[1]).unsqueeze(0)

            # Context features: home flag, ELOs
            matchup_elos = all_elos[
                (all_elos['SEASON_YEAR'] == convert_int_season_to_str(SEASON)) &
                (all_elos['TEAM_ABBREVIATION'].isin(list(matchup)))
            ]
            if len(matchup_elos) < 2:
                continue
            elos = [int(matchup_elos[matchup_elos['TEAM_ABBREVIATION'] == t]['elo_before_game'].iloc[0]) for t in matchup]
            context_data = [IS_HOME, elos[0], elos[1]]
            context_tensor = torch.tensor([context_data], dtype=torch.float32)

            # Label: did teamA win more often than teamB?
            h2h = get_head_to_head_win_pct(regular_games_total, matchup, season=SEASON)
            win_pct = h2h[matchup[0]]
            if win_pct is None:
                continue
            label = torch.tensor([[1.0 if win_pct > 0.5 else 0.0]])

            # Training step
            output = model(team1_tensor, team2_tensor, context_tensor)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_games += 1
        except Exception as e:
            print(f"Skipping matchup {matchup} due to error: {e}")

    print(f"[Epoch {epoch + 1}] Avg Loss: {total_loss / max(num_games,1):.4f} | Matchups trained on: {num_games}")

# ---------------- SAVE MODEL ---------------------
torch.save(model.state_dict(), "nba_matchup_model.pth")
print("Model saved as nba_matchup_model.pth")

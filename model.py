import torch
import torch.nn as nn
import torch.nn.functional as F

class FullPlayerTeamMatchupModel(nn.Module):
    def __init__(self, player_feat_dim, num_players, context_dim, hidden_dim=64):
        super().__init__()
        self.player_feat_dim = player_feat_dim
        self.num_players = num_players
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * num_players * hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, teamA_players, teamB_players, context):
        B, N, F = teamA_players.shape
        if N != self.num_players:
            raise ValueError(f"Input num_players {N} != model.num_players {self.num_players}")
        teamA_encoded = self.player_encoder(teamA_players.view(-1, F)).view(B, N, -1)
        teamB_encoded = self.player_encoder(teamB_players.view(-1, F)).view(B, N, -1)
        teamA_flat = teamA_encoded.view(B, -1)
        teamB_flat = teamB_encoded.view(B, -1)
        combined = torch.cat([teamA_flat, teamB_flat, context], dim=1)
        return self.final_mlp(combined)

import torch
import torch.nn as nn

class PlayerImpactMatchupModel(nn.Module):
    """
    Predicts P(team1 wins) based on:
    - Player impact values (per team)
    - Team ELOs
    - Home/away indicator

    Context vector shape: [is_home_flag, elo_team1, elo_team2]
    Player input shape: [batch_size, num_players, 1]  # impact per player
    """
    def __init__(self, num_players: int, context_dim: int = 3, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.num_players = num_players

        self.player_impact_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(2 * num_players * hidden_dim + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Predict win probabilities for both teams
        )

    def forward(self, team1_players, team2_players, context):
        B, N, F = team1_players.shape
        assert F == 1, "Expected impact scalar per player"
        assert N == self.num_players

        t1_encoded = self.player_impact_encoder(team1_players.view(-1, 1)).view(B, N, -1).flatten(start_dim=1)
        t2_encoded = self.player_impact_encoder(team2_players.view(-1, 1)).view(B, N, -1).flatten(start_dim=1)

        x = torch.cat([t1_encoded, t2_encoded, context], dim=1)
        return self.final_mlp(x)  # logits output

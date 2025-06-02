import pandas as pd
from typing import Dict, Tuple
import torch 
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

teams = ['DAL','MIL','ATL','DEN','HOU','IND','OKC','CHI','ORL','BOS','DET','NYK',
         'CHA','LAL','SAC','MIA','LAC','GSW','POR','MIN','WAS','BKN','MEM','SAS',
         'PHX','NOP','UTA','TOR','PHI','CLE']
         
regular_games_total = pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_totals_2010_2024.csv")
regular_season_all_parts = pd.concat([
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_1.csv"),
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_2.csv"),
    pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_3.csv")
])
def convert_int_season_to_str(season):
    if isinstance(season, int):
        return f"{season}-{season%2000 +1 :02d}" 
    return season

def convert_min_to_float(min_str):
    if isinstance(min_str, str) and ':' in min_str:
        mins, secs = map(int, min_str.split(':'))
        return mins + secs / 60
    return 0.0  # handle empty or malformed entries

def getMatchAndPlayerStats(game,player,season=None,teamname=None,filterFields=[]):
    """
    Function to get the average points of a team in a season
    :param teamname: team name :List[str]
    :param season: season: str or int or None (for all seasons) 
    :return: average points of the team 
    """
    season=convert_int_season_to_str(season)
    playerScores = player[player['minutes'].notna()].copy()
    playerScores['minutesParsed'] = playerScores['minutes'].apply(convert_min_to_float)
    game.loc[:,'WL'] = game['WL'].replace({'W': 1, 'L': 0}).infer_objects(copy=False)
    gamePlayer=game.merge(playerScores, how='inner', left_on=['GAME_ID','TEAM_ABBREVIATION'], right_on=['gameId','teamTricode'])
    # add a collumn to count the number of games played by each player
    aggregation= gamePlayer.groupby(['personName','teamTricode','season_year']).agg(
        {
            'WL': 'sum',
            'plusMinusPoints':'mean',
            'minutesParsed': 'mean',
            'points': 'mean',
            'fieldGoalsPercentage': 'mean',
            'threePointersPercentage': 'mean',
            'reboundsTotal': 'mean',
            'foulsPersonal': 'mean',
            'turnovers': 'mean',
            'fieldGoalsMade': 'mean',
            'fieldGoalsAttempted': 'mean',
            'steals':'mean',
        }
    ).reset_index()
    aggregation['gamesPlayed'] = gamePlayer.groupby(['personName','teamTricode','season_year'])['gameId'].count().reset_index(drop=True)

    if season is not None:
        aggregation = aggregation[aggregation['season_year'] == season]
    if teamname is not None:
        aggregation = aggregation[aggregation['teamTricode'].isin(teamname)]
    if len(filterFields)>0:
        aggregation.filter(items=filterFields)
    aggregation['winPercentage'] = aggregation['WL'] / aggregation['gamesPlayed'] 
    return aggregation.reset_index(drop=True)


def getMatchupByTeamBySeason(scores,matchup,season=False):
    """
    Function to get the matchup of a team in a season
    :param team_tag: team tag
    :optional param season: season to filter the data by season 
    :return: matchup of the team in the season
    """
    teams=scores.filter(items=['SEASON_YEAR','TEAM_ABBREVIATION','GAME_ID','MATCHUP','WL'])
    teams.loc[:,'WL'] = teams['WL'].replace({'W': 1, 'L': 0}).infer_objects(copy=False)
    if season is not False:
        teams=teams[teams['SEASON_YEAR']==convert_int_season_to_str(season)]
    mathcup_tag=matchup[0]+" vs. "+matchup[1]
    matchup_inverse_tag=matchup[1]+" vs. "+matchup[0]
    teams['MATCHUP_STANDARD'] = teams['MATCHUP'].str.replace("@", "vs.")
    teams=pd.concat([teams[teams['MATCHUP_STANDARD'] ==  mathcup_tag],teams[teams['MATCHUP_STANDARD'] ==  matchup_inverse_tag]],ignore_index=True)
    teams['IS_HOME'] = teams['MATCHUP'].str.contains('@')
    return teams.filter(items=['SEASON_YEAR','TEAM_ABBREVIATION','GAME_ID','IS_HOME','MATCHUP_STANDARD','WL'])

def aggregate_matchup_data(df, matchup):
    """
    Combines the df, which contains game data, with the specified matchup.
    The matchup is a tuple of two team abbreviations (e.g., ('LAL', 'BOS')).
    If a team abbreviation is '*', it acts as a wildcard and matches any team.
    The function returns a DataFrame with the following columns:
    - GAME_ID
    - SEASON_YEAR
    - HOME_TEAM
    - AWAY_TEAM
    - HOME_WL
    - AWAY_WL
    - MATCHUP_STANDARD
    :param df: DataFrame containing game data with columns:
               - GAME_ID
               - SEASON_YEAR
               - TEAM_ABBREVIATION
               - MATCHUP_STANDARD
               - IS_HOME (boolean indicating if the team is home)
               - WL (win/loss record)
    """
    team1_abbr, team2_abbr = matchup

    # Handle wildcards: build valid team list
    if team1_abbr == "*":
        teams1 = df['TEAM_ABBREVIATION'].unique()
    else:
        teams1 = [team1_abbr]

    if team2_abbr == "*":
        teams2 = df['TEAM_ABBREVIATION'].unique()
    else:
        teams2 = [team2_abbr]

    # Filter for games where team is in either team1 or team2
    filtered = df[
        df['TEAM_ABBREVIATION'].isin(set(teams1) | set(teams2))
    ]

    # Build dynamic mask for MATCHUP_STANDARD column
    mask = filtered['MATCHUP_STANDARD'].apply(
        lambda x: any(f"{a} vs. {b}" in x or f"{b} vs. {a}" in x for a in teams1 for b in teams2 if a != b or team1_abbr == "*" or team2_abbr == "*")
    )
    filtered = filtered[mask]

    # Label as HOME or AWAY
    filtered = filtered.copy()
    filtered['HOME_OR_AWAY'] = filtered['IS_HOME'].map({True: 'HOME', False: 'AWAY'})

    # Pivot to get HOME and AWAY stats side by side
    pivoted = filtered.pivot(index='GAME_ID', columns='HOME_OR_AWAY')

    # Flatten multi-index columns
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted = pivoted.reset_index()

    # Select desired columns
    result = pivoted[[
        'GAME_ID',
        'SEASON_YEAR_HOME',
        'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
        'WL_HOME', 'WL_AWAY',
        'MATCHUP_STANDARD_HOME'
    ]].rename(columns={
        'SEASON_YEAR_HOME': 'SEASON_YEAR',
        'TEAM_ABBREVIATION_HOME': 'HOME_TEAM',
        'TEAM_ABBREVIATION_AWAY': 'AWAY_TEAM',
        'WL_HOME': 'HOME_WL',
        'WL_AWAY': 'AWAY_WL',
        'MATCHUP_STANDARD_HOME': 'MATCHUP_STANDARD'
    })

    return result

def playerMatchUpIntersection(regular_season_all_parts, matchup_games, players_df: pd.DataFrame):
    """
    For each player in the given DataFrame, calculate their win % in a specific matchup.

    :param regular_season_all_parts: DataFrame of all box scores
    :param matchup_games: DataFrame of games in a specific matchup
    :param players_df: DataFrame with at least ['personName', 'teamTricode', 'PlayerImpactM1', ...]
    :return: DataFrame with matchup stats per player
    """
    matchup_games = matchup_games.rename(columns={"GAME_ID": "gameId"})
    rows = []

    for _, row in players_df.iterrows():
        name = row['personName']
        team = row['teamTricode']
        metrics = row.drop(['personName', 'teamTricode']).to_dict()

        # Filter for this player's stats
        player_stat = regular_season_all_parts[
            (regular_season_all_parts['personName'] == name) &
            (regular_season_all_parts['teamTricode'] == team) &
            (regular_season_all_parts['minutes'].notna())
        ][['season_year', 'gameId', 'teamTricode', 'personName', 'minutes']]

        merged_df = pd.merge(player_stat, matchup_games, on='gameId', how='inner')
        if merged_df.empty:
            continue

        # Determine win
        def did_win(row):
            if row['teamTricode'] == row['HOME_TEAM']:
                return row['HOME_WL'] == 1
            elif row['teamTricode'] == row['AWAY_TEAM']:
                return row['AWAY_WL'] == 1
            return False

        merged_df['won'] = merged_df.apply(did_win, axis=1)

        total_games = len(merged_df)
        total_wins = merged_df['won'].sum()
        win_percentage = total_wins / total_games if total_games > 0 else 0

        # Collect result row
        result = {
            'personName': name,
            'teamTricode': team,
            'gamePlayed': total_games,
            'matchupWinPercentage': round(win_percentage, 3),
            'MATCHUP_STANDARD': merged_df['MATCHUP_STANDARD'].iloc[0]
        }
        result.update(metrics)

        rows.append(result)

    return pd.DataFrame(rows)


def calculate_elo_rating(df, initial_elo=1500, k=20):
    """
    Calcula o Elo Rating de cada equipa ao longo da época regular.
    Adiciona uma nova coluna 'elo_before_game' ao DataFrame com o valor de Elo antes de cada jogo.
    
    :param df: DataFrame com os jogos (incluindo GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MATCHUP, WL)
    :param initial_elo: Elo inicial para todas as equipas
    :param k: fator de aprendizagem (K-factor)
    :return: DataFrame com a coluna 'elo_before_game' adicionada
    """

    # Ordenar por data para garantir cronologia
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')

    # Extrair equipas em casa e fora
    def get_home_away(row):
        if "vs." in row['MATCHUP']:
            return row['TEAM_ABBREVIATION'], row['MATCHUP'].split('vs. ')[1]
        elif "@" in row['MATCHUP']:
            return row['MATCHUP'].split('@ ')[1], row['TEAM_ABBREVIATION']
        return None, None

    df[['HOME_TEAM', 'AWAY_TEAM']] = df.apply(get_home_away, axis=1, result_type='expand')
    df = df.dropna(subset=['HOME_TEAM', 'AWAY_TEAM', 'WL'])

    # Inicializar ratings
    elo_ratings = {}
    elo_gamewise = []

    for game_id in df['GAME_ID'].unique():
        game_df = df[df['GAME_ID'] == game_id]

        if len(game_df) != 2:
            continue

        team1 = game_df.iloc[0]
        team2 = game_df.iloc[1]

        t1_tag = team1['TEAM_ABBREVIATION']
        t2_tag = team2['TEAM_ABBREVIATION']

        r1 = elo_ratings.get(t1_tag, initial_elo)
        r2 = elo_ratings.get(t2_tag, initial_elo)

        result1 = 1 if team1['WL'] == 'W' else 0
        result2 = 1 - result1

        elo_gamewise.append({
            'GAME_ID': game_id,
            f"{t1_tag}_elo_before": r1,
            f"{t2_tag}_elo_before": r2
        })

        # Função auxiliar
        def expected(ra, rb):
            return 1 / (1 + 10 ** ((rb - ra) / 400))

        e1 = expected(r1, r2)
        e2 = expected(r2, r1)

        elo_ratings[t1_tag] = r1 + k * (result1 - e1)
        elo_ratings[t2_tag] = r2 + k * (result2 - e2)

    # Criar DataFrame auxiliar
    elo_df = pd.DataFrame(elo_gamewise)

    # Função para recuperar o elo da equipa antes do jogo
    def get_elo(row):
        team = row['TEAM_ABBREVIATION']
        game_id = row['GAME_ID']
        match = elo_df[elo_df['GAME_ID'] == game_id]
        if not match.empty:
            return match.iloc[0].get(f"{team}_elo_before", None)
        return None

    df['elo_before_game'] = df.apply(get_elo, axis=1)
    return df

def get_head_to_head_win_pct(df: pd.DataFrame, matchup: tuple, season=None):
    team1_abbr, team2_abbr = matchup

    if season is not None:
        season = convert_int_season_to_str(season)
        df = df[df["SEASON_YEAR"] == season].copy()

    # Keep original MATCHUP column to extract home/away info
    df['MATCHUP_STANDARD'] = df['MATCHUP'].str.replace("@", "vs.", regex=False)

    matchup_tag = f"{team1_abbr} vs. {team2_abbr}"
    matchup_inverse_tag = f"{team2_abbr} vs. {team1_abbr}"

    # Filter relevant games
    df = df[df['MATCHUP_STANDARD'].isin([matchup_tag, matchup_inverse_tag])].copy()

    # Drop duplicate games by GAME_ID
    df = df.drop_duplicates(subset='GAME_ID', keep='first')

    if df.empty:
        return {
            "TEAM_A": team1_abbr,
            "TEAM_B": team2_abbr,
            f"{team1_abbr}_home_win_pct": None,
            f"{team1_abbr}_away_win_pct": None,
            f"{team2_abbr}_home_win_pct": None,
            f"{team2_abbr}_away_win_pct": None,
            "total_games": 0
        }

    # Determine home and away teams from the original MATCHUP string
    def get_home_away(row):
        parts = row['MATCHUP'].split(' ')
        if '@' in row['MATCHUP']:
            away, home = parts[0], parts[2]
        else:
            away, home = parts[2], parts[0]
        return pd.Series([home, away])

    df[['HOME_TEAM', 'AWAY_TEAM']] = df.apply(get_home_away, axis=1)

    # Helper function to compute win percentage
    def compute_win_pct(team, location):
        if location == "home":
            games = df[df['HOME_TEAM'] == team]
            wins = games[(games['TEAM_ABBREVIATION'] == team) & (games['WL'] == 'W')]
        else:
            games = df[df['AWAY_TEAM'] == team]
            wins = games[(games['TEAM_ABBREVIATION'] == team) & (games['WL'] == 'W')]
        total = len(games)
        return round(len(wins) / total, 3) if total > 0 else None
    def compute_total_win_pct(team):
        games = df[df['TEAM_ABBREVIATION'] == team]
        wins = games[games['WL'] == 'W']
        total = len(games)
        return round(len(wins) / total, 3) if total > 0 else None

    return {
        "TEAM_A": team1_abbr,
        "TEAM_B": team2_abbr,
        f"{team1_abbr}_home_win_pct": compute_win_pct(team1_abbr, "home"),
        f"{team1_abbr}_away_win_pct": 1 - compute_win_pct(team2_abbr, "home"),
        f"{team2_abbr}_home_win_pct": compute_win_pct(team2_abbr, "home"),
        f"{team2_abbr}_away_win_pct": 1 - compute_win_pct(team1_abbr, "home"),
        f"{team1_abbr}_total_win_pct": compute_total_win_pct(team1_abbr),
        "total_games": len(df)
    }

def predict_matchup(model, playersStats, all_elos, matchup, season, is_home, player_count):
    regular_games_total = pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_totals_2010_2024.csv")
    model.eval()

    team1, team2 = matchup
    season_str = convert_int_season_to_str(season)
    
    # Get Elo values
    matchUp_elos = all_elos[
        (all_elos['SEASON_YEAR'] == season_str) &
        (all_elos['TEAM_ABBREVIATION'].isin([team1, team2]))
    ]
    elo_teams = [int(matchUp_elos[matchUp_elos['TEAM_ABBREVIATION'] == t]['elo_before_game'].iloc[0]) for t in matchup]
    elos = [(e - 1300) / 500 for e in elo_teams]  # ELO scaling
    # Player features
    player_features = ['playerImpact']
    players = [
        playersStats[
            (playersStats['teamTricode'] == t) & 
            (playersStats['season_year'] == season_str)
        ].filter(items=player_features).to_dict(orient="list")
        for t in matchup
    ]
    players = [list(zip(*(playerTeam[f] for f in player_features))) for playerTeam in players]
    teamsData = {t: np.array(players[i], dtype=float) for i, t in enumerate(matchup)}

    # Use fixed player_count (from training/model), but check if data has enough players:
    for t in matchup:
        if len(teamsData[t]) < player_count:
            raise ValueError(f"Not enough players for team {t} in prediction: has {len(teamsData[t])}, requires {player_count}")

    # Tensors
    team1_tensor = torch.tensor(teamsData[team1][:player_count]).unsqueeze(0).float()
    team2_tensor = torch.tensor(teamsData[team2][:player_count]).unsqueeze(0).float()
    h2h = get_head_to_head_win_pct(regular_games_total, matchup, season=season)
    contextData = [is_home, elos[0], elos[1]]#,home_win_percetage_t1,away_win_percetage_t2]
    context_tensor = torch.tensor([contextData], dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        pred = model(team1_tensor, team2_tensor, context_tensor)

    team1_prob = float(pred[0][0].item())
    team2_prob = float(pred[0][1].item())
    return team1_prob,team2_prob

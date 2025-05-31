import itertools
import pandas as pd
from IPython.display import display
import sklearn
import json 
from xgboost import XGBRegressor
from utils import getMatchAndPlayerStats, getMatchupByTeamBySeason,aggregate_matchup_data,playerMatchUpIntersection,calculate_elo_rating,get_head_to_head_win_pct
pd.set_option('future.no_silent_downcasting',True)
DATA_FOLDER="./datasets/DATA_AGGREGATIONS/"
SEASON=2023

NUM_GAMES=82
teams=['DAL','MIL','ATL','DEN','HOU','IND','OKC','CHI','ORL','BOS','DET','NYK'
,'CHA','LAL','SAC','MIA','LAC','GSW','POR','MIN','WAS','BKN','MEM','SAS'
,'PHX','NOP','UTA','TOR','PHI','CLE']
all_possible_matchups=list(itertools.combinations(teams, 2))
print("Loading regular season data...")
regular_games_total=pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_totals_2010_2024.csv",delimiter=',',header=0)
regular_season_all_parts=pd.concat([
        pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_1.csv",delimiter=',',header=0),
        pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_2.csv",delimiter=',',header=0),
        pd.read_csv("./datasets/NBA_DATA_2010_2024/regular_season_box_scores_2010_2024_part_3.csv",delimiter=',',header=0)])

playersStats=pd.read_csv(DATA_FOLDER+"/playerStats.csv")
# get all the teams matchup data
print("Getting matchup data...")
all_elos = pd.read_csv(DATA_FOLDER+"gamesAndEloStats.csv")

print("Joining playerstats with team elos...")
playersStats=playersStats.merge(right=all_elos,left_on="teamTricode",right_on="TEAM_ABBREVIATION",how="inner")
display(playersStats)


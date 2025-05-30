import itertools
import pandas as pd
from IPython.display import display
import sklearn
import json 
from xgboost import XGBRegressor
from utils import getMatchAndPlayerStats, getMatchupByTeamBySeason,aggregate_matchup_data,playerMatchUpIntersection
pd.set_option('future.no_silent_downcasting',True)



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
print("Regular season data loaded")
# get all the players data
# get the players stats for the regular season
playersStats=getMatchAndPlayerStats(regular_games_total, regular_season_all_parts,season=2023,filterFields=["personName","teamTricode","WL","minutesParsed","points","fieldGoalsPercentage","threePointersPercentage","reboundsTotal","foulsPersonal","turnovers","fieldGoalsMade","fieldGoalsAttempted","steals","gamesPlayed"])

display(playersStats)

print("Scaling player stats...")
scaler = sklearn.preprocessing.StandardScaler()
playersStats_scaled = scaler.fit_transform(playersStats.drop(columns=['personName', 'teamTricode']))

print("Loading the model...")
model1 = XGBRegressor()
model1.load_model("./models/xgb_tuned.json")
print("Making predictions...")
display(len(playersStats_scaled[0]))
y_pred_loaded = model1.predict(playersStats_scaled)
# append the impact stat to the players stats
playersStats["PlayerImpact"]=y_pred_loaded
display(playersStats)
# get all the teams matchup data
print("Getting matchup data...")
mt=pd.DataFrame([])
for matchup in all_possible_matchups:
    mt = pd.concat([mt,getMatchupByTeamBySeason(regular_games_total,matchup,season=2023)],ignore_index=True)
display(mt)
print("Intersecting with player data...")
matchup_games=aggregate_matchup_data(mt,("DAL","GSW"))
playerMatchupStats=playerMatchUpIntersection(regular_season_all_parts,matchup_games,playersStats)
if playerMatchupStats is not None:
    display(playerMatchupStats)

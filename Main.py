import requests
import json
from math import sqrt
from math import floor
from math import ceil
import csv
import pandas as pd

#The starting year of the season (by default this is the 2017-2018 season).
#The earliest year that will work is 2010.
YEAR = "2018"

##Note periods go 1, 2, 3 with 4 OT and 5 SO##
#Constants for the type of games played
PRESEASON = "01"
REGULARSEASON = "02"
PLAYOFFS = "03"
ALLSTAR = "04"

goals = []
nongoals = []

goal_shot = [] ###xcoord, ycoord, shottype,
nongoal_shot = []

def distance(x,y):
   return (sqrt((89-x)**2+(y)**2))

#Loop through every game in a season (1271 for 31 teams)
with open("2018NHLShotInfo.csv", 'a') as f:
   fieldnames = ["Shot", "X", "Y", "Shot_Type", "Shooter", "Team", "Home_Away", "Period", "Year"]
   writer = csv.DictWriter(f, delimiter = ",", fieldnames=fieldnames)
   writer.writeheader()
   for i in range(1,1272):
      try:
         #Get the API url
         url = "https://statsapi.web.nhl.com/api/v1/game/" + YEAR + REGULARSEASON + str("%04d" %(i,)) + "/feed/live"
         response = requests.get(url)
         home = response.json()["gameData"]["teams"]["home"]["triCode"]
         away = response.json()["gameData"]["teams"]["away"]["triCode"]
         for j in range(1000):
            try:
               if response.json()["liveData"]["plays"]["allPlays"][j]["result"]["event"] == "Goal":
                  gteam = response.json()["liveData"]["plays"]["allPlays"][j]["team"]["triCode"]
                  gperiod = response.json()["liveData"]["plays"]["allPlays"][j]["about"]["period"]
                  goalx = float(response.json()["liveData"]["plays"]["allPlays"][j]["coordinates"]["x"])
                  goaly = float(response.json()["liveData"]["plays"]["allPlays"][j]["coordinates"]["y"])
                  gshot_type = response.json()["liveData"]["plays"]["allPlays"][j]["result"]["secondaryType"]
                  #desc = response.json()["liveData"]["plays"]["allPlays"][j]["result"]["description"]
                  gshooter = response.json()["liveData"]["plays"]["allPlays"][j]["players"][0]["player"]["fullName"]
                  if gteam == home:
                     home_team = 1
                  else:
                     home_team = 0
                  #goals.append(distance(goalx,goaly))
                  #goal_shot.append([goalx, goaly, gshot_type, desc])
                  writer.writerow({"Shot":"Goal", "X":goalx, "Y" : goaly, "Shot_Type" : gshot_type, "Shooter" : gshooter, "Team" : gteam, "Home_Away" : home_team, "Period" : gperiod , "Year" : YEAR})
               if response.json()["liveData"]["plays"]["allPlays"][j]["result"]["event"] == "Shot":
                  nongoalx = float(response.json()["liveData"]["plays"]["allPlays"][j]["coordinates"]["x"])
                  nongoaly = float(response.json()["liveData"]["plays"]["allPlays"][j]["coordinates"]["y"])
                  ngteam = response.json()["liveData"]["plays"]["allPlays"][j]["team"]["triCode"]
                  ngperiod = response.json()["liveData"]["plays"]["allPlays"][j]["about"]["period"]
                  ngshot_type = response.json()["liveData"]["plays"]["allPlays"][j]["result"]["secondaryType"]
                  #ngdesc = response.json()["liveData"]["plays"]["allPlays"][j]["result"]["description"]
                  ngshooter = response.json()["liveData"]["plays"]["allPlays"][j]["players"][0]["player"]["fullName"]
                  if ngteam == home:
                     home_team = 1
                  else:
                     home_team = 0
                  #nongoals.append(distance(nongoalx,nongoaly))
                  #nongoal_shot.append([nongoalx, nongoaly, ngshot_type, ngdesc])
                  writer.writerow({"Shot" : "ngshot", "X" : nongoalx, "Y" : nongoaly, "Shot_Type" : ngshot_type, "Shooter" : ngshooter, "Team" : ngteam, "Home_Away" : home_team, "Period" : ngperiod, "Year" : YEAR})
            except IndexError:
               break
      except KeyError:
         continue

#shotCounter = [0 for i in range(100)]
#goalCounter = [0 for i in range(100)]

#for i in range(len(goals)):
#   goalCounter[int(floor(goals[i]))] = goalCounter[int(floor(goals[i]))] + 1
#for i in range(len(nongoals)):
#   shotCounter[int(floor(nongoals[i]))] = shotCounter[int(floor(nongoals[i]))] + 1

#print('Distance(ft)\tShooting Percentage\n')
#for i in range(len(goalCounter)):
#   if goalCounter[i] + shotCounter[i] != 0:
#      print(str(i+1) + '\t\t' + str(goalCounter[i]) + '/' + str(goalCounter[i]+shotCounter[i]) + ' = ' + \
#         str(float(goalCounter[i])/float((goalCounter[i]+shotCounter[i]))))


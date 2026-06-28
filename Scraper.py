import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np

# TO DO: need to create a csv to use for the entire data spit out for use with these functions
def league_year_scrape( league, year):
    hockey_soup = BeautifulSoup("http://www.hockeydb.com/ihdb/stats/leagues.html", features="html.parser")
    ###For now we do current leagues
    #Professional: National Hockey League (NHL)
    #Minor Professional: AHL, ECHL, FHL, LNAH, SPHL
    #Major Junior: OHL, QMJHL, WHL
    #College: CWUAA, OUAA, NCAA, AHA, Big-10, CCC, ECAC, H-East, MASCAC, MIAC, NCHC, NESCAC, NEHC, NE-10, NCHA, SUNYAC, UCHC, WCHA, WIAC
    #Canada Junior A: AJHL, BCHL, CCHL, MHL, NOJHL, OJHL, QJAAAHL, SJHL, SIJHL
    ###From league search page, have a lookup for league nickname to technical name, then lookup on table and click on correct link
    League_Dict = {"OHL":"Ontario Hockey League", "NHL": "National Hockey League", "AHL": "American Hockey League", "ECHL":"ECHL", "FHL":"Federal Prospects Hockey League", "LNAH":"Ligue Nord-Americaine de Hockey", "SPHL":"Southern Professional Hockey League", "QMJHL":"Quebec Major Junior Hockey League", "WHL":"Western Hockey League", "CWUAA":"CIS - Canada West Universities Athletic Assn", "OUAA":"CIS - Ontario University Athletic Association", "NCAA":"National Collegiate Athletic Association", "AHA":"NCAA - Atlantic Hockey Association - Div. 1", "Big-10":"NCAA - Big 10 - Div. 1", "CCC":"NCAA - Commonwealth Coast Conference", "ECAC":"NCAA - ECAC - Div. 1", "H-East":"NCAA - Hockey East - Div. 1", "MASCAC":"NCAA - MASCAC", "MIAC":"NCAA - Minnesota Intercollegiate Athletic Conf.", "NCHC":"NCAA - National Collegiate Hockey Conf. - Div. 1", "NESCAC":"NCAA - NESCAC", "NEHC":"NCAA - New England Hockey Conference", "NE-10":"NCAA - Northeast 10", "NCHA":"NCAA - Northern Collegiate Hockey Association", "SUNYAC":"NCAA - SUNYAC", "UNHC":"NCAA - United Collegiate Hockey Conference", "WCHA":"NCAA - Western Collegiate Hockey Assn. - Div. 1", "WIAC":"NCAA - Wisconsin Intercollegiate Athletic Conf.", "AJHL":"Alberta Junior Hockey League", "BCHL":"British Columbia Hockey League", "MHL":"Maritime Hockey League", "NOJHL":"Northern Ontario Junior Hockey League", "OJHL":"Ontario Junior Hockey League", "QJAAAHL":"Quebec Junior Hockey League", "SIJHL":"Superior International Jr Hockey League"}
    hockey_table = hockey_soup.findAll('table')
    leagues = hockey_table.findAll('td', {'class' : "l"})
    league_name = []
    league_link = []
    for index, league in enumerate(leagues):
        league_link.append(league.findAll('a')[index]["href"])
        league_name.append(league.findAll('a')[index].text)

    if League_Dict(league) in league_name:
        link = league_link[league_name.index(League_Dict(league))]
    else:
        return("The league you selected is either wrong or not implimented. Refer to the readme to see the leagues available")

    league_soup = BeautifulSoup(link, features = 'html.parser')
    league_table = league_soup.findAll('table')
    years = league_table.findAll('tr', {'class' : "shade4"})
    year_num = []
    year_link = []
    for index, year in enumerate(years):
        year_link.append(year.findAll('a')[index]["href"])
        year_num.append(year.findAll('a')[index].text)
    year_strip = int(str(year)[-2:]) + 1
    year_check = str(year) + "-" + str(year_strip)
    if year_check in year_num:
        link = year_link[year_num.index(year_check)]
    else:
        return("The year you selected is either wrong, not implemented or otherwise incorrect. Refer to the readme to see the years available for your chosen league")

    team_soup = BeautifulSoup(link, features = 'html.parser')
    team_table = team_soup.findAll('table')
    teams_body = team_table.findAll('tbody')
    teams = []
    team_links = []
    for index, team in enumerate(teams):
        teams.append(team_table.findAll('a')[index].text)
        team_links.append(team_table.findAll('a')[index]["href"])
    for index, team in enumerate(teams):
        team_scrape(team_links[index], file_name)

def team_scrape(link, file_name):
    player_soup = BeautifulSoup( link, features = 'html.parser')
    player_table = player_soup.findAll('table').findAll('tbody')
    players = player_table.findAll('tr')
    player_data = []
    for index, player in enumerate(players):# '#	Player Name	Pos.	GP	G	A	Pts	PIM	GP	G	A	Pts	PIM	Birthplace	Age' [15]
        player_data.append(players.findAll('td')[index].text)


def player_scrape(link, file_name):
    #implimented later. Will find height, weight and DOB

if __name__ == "__main__":
    league_year_scrape(OHL, 2019)
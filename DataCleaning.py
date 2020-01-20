import requests
import json
from math import sqrt
from math import floor
from math import ceil
import csv
import pandas as pd

header = ["Goal", "X", "Y", "Shot", "Shooter", "Desc", "Dist", "indeces"]
data = pd.read.csv("2018NHLShootingInfo.csv")


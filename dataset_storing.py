"""
Stores streamlined code for cal_fire_temps dataset so it can be imported into the
Jupyter Notebooks for the models
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

calfires = gpd.read_file("fire21_2.gdb")

calfires_trim = calfires.drop(labels=["STATE", "FIRE_NAME", "AGENCY", "UNIT_ID", "INC_NUM", "COMMENTS", "REPORT_AC", "C_METHOD", "OBJECTIVE", "FIRE_NUM", "COMPLEX_NAME", "COMPLEX_INCNUM", "CONT_DATE"], axis=1)
calfires_trim = calfires_trim.dropna()

def date_trim(date):
    return date[:10]

calfires_trim.ALARM_DATE = calfires_trim.ALARM_DATE.map(date_trim)

calfires_trim["ALARM_DATE"] = pd.to_datetime(calfires_trim["ALARM_DATE"], format="%Y-%m-%d", errors="coerce")

cal_temps_csv = open("cal_temps.csv")
cal_temps = pd.read_csv(cal_temps_csv)

cal_temps_trim = cal_temps.drop(labels=["Unnamed: 0", "AverageTemperatureUncertainty", "Country", "InCal"], axis=1)
cal_temps_trim.dropna(inplace=True)

cal_temps_trim["dt"] = pd.to_datetime(cal_temps_trim["dt"], format="%Y-%m-%d")

def find_temps(date: pd.Timestamp):
    mod_date = pd.Timestamp(f"{date.year}-{date.month}-01")
    date_temps = cal_temps_trim.loc[cal_temps_trim["dt"] == mod_date]
    temps_list = []
    for i in date_temps.index:
        temps_list.append(date_temps["AverageTemperature"][i])
    
    return temps_list

def get_dates_list(date):
    return pd.date_range(end=date, freq="M", periods=12)

def assign_temperatures(fire):
    date = fire["ALARM_DATE"]
    try:
        for i in range(1, 13):
            fire[f"{i}_MONTHS_BEFORE"] = find_temps(get_dates_list(date)[i - 1])
    except:
        print(fire)
    return fire

cal_fire_temps = calfires_trim.apply(lambda row: assign_temperatures(row), axis=1)

global null_indices
null_indices = []

for i in cal_fire_temps.index:
    for j in range(1, 13):
        if cal_fire_temps[f"{j}_MONTHS_BEFORE"][i] == []:
            if not i in null_indices:
                null_indices.append(i)

cal_fire_temps = cal_fire_temps.drop(null_indices)

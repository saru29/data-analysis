import time
import sns as sns
import os
from datetime import datetime
from heatmap import heatmap, corrplot
from google.cloud import bigquery
from os.path import exists as file_exists
from google.oauth2 import service_account
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import seaborn as sn
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pandas.io import gbq  # to communicate with Google BigQuery

##################
# M.TELLEY
##################

# Set up connection with BigQuery Project tp retrive data
credentials = service_account.Credentials.from_service_account_file('XXXXXXXXXXXXXXXXXXXXXXXX.json')
project_id = 'XXXXXXXXXXXXXXXXXX'
client = bigquery.Client(credentials=credentials, project=project_id)
print("Client creating using default project: {}".format(client.project))

def filterDF(x,y,z):
    dfx = x[x[y] == z]

##################
# Get Data
# Data frame created and stored on local - DELETE the data.csv file to return different data results.
runit = False
if file_exists('data.csv'):
    # if runit:
    print("File exists")
    df1 = pd.read_csv("data.csv")
    df1 = df1[df1['date'] == "2022-11-19"]
    print("Loading Data...")
else:
    print("Attempting to write query results to local")
    sql_query = """
        SELECT * FROM `BIGQUERYDB.Master_Fitness_Data.master-fitness-user1`
        ORDER BY userID, tiemstamp_AEST
    """
    ##################
    # Organise script data for dataframe

    job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
    results = client.query(sql_query, job_config=job_config)
    xx = 0
    r = 0
    df1 = pd.DataFrame(
        columns=['userID', 'date', 'tiemstamp_AEST', 'distance', 'distance_per_second', 'enhanced_speed', 'power', 'wpkg', 'FTP_perct',
                 'cadence', 'grade', 'enhanced_altitude', 'heart_rate', 'max_heart_rate_perct', 'day_of_week', 'hour'])

    for row in results:
        userID = row['userID']
        date = row['date']
        tiemstamp_AEST = row['tiemstamp_AEST']
        distance = float((row['distance']))
        distance_per_second = row['distance_per_second']
        enhanced_speed = row['enhanced_speed']
        power = row['power']
        wpkg = row['wpkg']
        FTP_perct = row['FTP_perct']
        cadence = row['cadence']
        grade = row['grade'],
        enhanced_altitude = row['enhanced_altitude']
        heart_rate = int(row['heart_rate'])
        max_heart_rate_perct = row['max_heart_rate_perct']
        day_of_week = int(row['day_of_week'])
        hour = int(row['hour'])
        df1.loc[r] = [userID,
                      date,
                      tiemstamp_AEST,
                      distance,
                      enhanced_speed,
                      distance_per_second,
                      power,
                      wpkg,
                      FTP_perct,
                      cadence,
                      grade,
                      enhanced_altitude,
                      heart_rate,
                      max_heart_rate_perct,
                      day_of_week,
                      hour]
        r += 1

    df1.to_csv("data.csv")
    df1 = pd.read_csv("data.csv")
    if file_exists('data.csv'):
        print("File Created")
    else:
        if len(df1.index) < 1:
            print("File created, but no data rows: ", len(df1.index))
        else:
            print("Failed to created file")

    print(df1.head())

##################
# Correlate numeric data attributes
# Reformat DF to have only numeric data attributes
dfCor = df1[['distance_per_second',
             'enhanced_speed',
             'power',
             'wpkg',
             'FTP_perct',
             'cadence',
             'grade',
             'enhanced_altitude',
             'heart_rate',
             'max_heart_rate_perct']]
matrix = dfCor.corr()
#print(matrix)
#corrplot(matrix, size_scale=500, marker='s')
matrix.to_csv("data_corr.csv")

# print(len(df1.index))

##################
# PLOT KEY Correlations
df1_temp = df1[['wpkg']].copy()
fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(df1_temp, bins = [0,1,2,3,4,5,6,7,8,9,10,11,12])
plt.show()
#print(df1_temp.head())
# Displaying the graph
#plt.show()

#################
# Data views - upskilling

y1 = df1["grade"].to_numpy()
x1 = df1["wpkg"].to_numpy()

y2 = df1["cadence"].to_numpy()
x2 = df1["wpkg"].to_numpy()

y3 = df1["heart_rate"].to_numpy()
x3 = df1["wpkg"].to_numpy()

y4 = df1["max_heart_rate_perct"].to_numpy()
x4 = df1["distance_per_second"].to_numpy()

y5 = df1["cadence"].to_numpy()
x5 = df1["enhanced_speed"].to_numpy()
# subplot(row, columns, place

plt.subplot(621)
plt.ylim(-5, 10)
plt.scatter(x1, y1, 0.5)
trend = np.polyfit(x1, y1, 1)
trendpoly = np.poly1d(trend)
plt.plot(x1, trendpoly(x1))
plt.ylabel('grade')
plt.xlabel('wpkg')
plt.grid(True)

plt.subplot(622)
plt.ylim(85, 120)
plt.scatter(x2, y2, 0.5)
trend = np.polyfit(x2, y2, 1)
trendpoly = np.poly1d(trend)
plt.plot(x2, trendpoly(x2))
plt.ylabel('cadence')
plt.xlabel('wpkg')
plt.grid(True)

plt.subplot(625)
plt.ylim(125, 205)
plt.scatter(x3, y3, 0.5)
trend = np.polyfit(x3, y3, 1)
trendpoly = np.poly1d(trend)
plt.plot(x3, trendpoly(x3))
plt.ylabel('heart_rate')
plt.xlabel('wpkg')
plt.grid(True)

plt.show()

new_df = df1[['tiemstamp_AEST','wpkg']].copy()
print(new_df.head())
x4 = new_df["tiemstamp_AEST"].to_numpy()
y4 = new_df["wpkg"].to_numpy()
plt.plot(x4, y4, color ='tab:blue')
plt.grid(True)
plt.show()

new_df1 = df1[['tiemstamp_AEST']].copy()
x = new_df1[["tiemstamp_AEST"]].to_numpy()
new_df2 = df1[['wpkg']].copy()
y = new_df2[["wpkg"]].to_numpy()


# Loading the dataset
dfx = pd.DataFrame(
        columns=['tiemstamp_AEST', 'wpkg'])
dfx.to_csv("data_x.csv")
data = pd.read_csv("data_x.csv")

# Y axis is price closing
watts_per_kg = data['wpkg']

#########################
# Create new dataframe wih just power and time step

df1 = pd.read_csv("data.csv")
df1 = df1[df1['date'] == "2022-11-19"]
dfy = df1[['power']]
nrow = (len(dfy))
indexV = []
for i in range(nrow):
    indexV.append(i+1)
dfx = pd.DataFrame(indexV,
        columns=['step'])




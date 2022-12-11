import time
from google.cloud import bigquery
import os
from google.oauth2 import service_account
import pandas as pd
from datetime import datetime
from heatmap import heatmap, corrplot
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff

import pandas as pd
from pandas.io import gbq  # to communicate with Google BigQuery

# Script-VERSION 1-COMMENTS TO BE DOCUMENT AND CODE TO BE CLEANED UP
# Contact Mark Telley for acccess credentials
credentials = service_account.Credentials.from_service_account_file('redbackoperationsdataai-XXXXXXXXXXXX.json')

project_id = 'redbackoperationsdataai'

client = bigquery.Client(credentials=credentials, project=project_id)

print("Client creating using default project: {}".format(client.project))

sql_query = """
    SELECT
        *
    FROM
      `redbackoperationsdataai.fitness_user_summary_features.feature-summary-filtered`
    WHERE avg_power IS NOT NULL AND enhanced_avg_speed IS NOT NULL
    --LIMIT 100
"""

job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
results = client.query(sql_query, job_config=job_config)
xx = 0

df1 = pd.DataFrame(columns=['userID',
                            'date_AEST',
                            'distance',
                            'enhanced_avg_speed',
                            'avg_power',
                            'avg_wpkg',
                            'avg_heart_rate',
                            'avg_max_heart_rate_perct',
                            'enhanced_duration',
                            'weekend',
                            'day_of_week',
                            'hour',
                            'day_cat'])

r = 0;
for row in results:
    userID = row['userID']
    date_AEST = datetime.strptime(str(row['date_AEST']), '%Y-%m-%d')
    date_AEST = date_AEST.date()
    distance = float((row['distance']))
    enhanced_avg_speed = float(row['enhanced_avg_speed'])
    avg_power = float(row['avg_power'])
    avg_wpkg = float(row['avg_wpkg'])
    avg_heart_rate = float(row['avg_heart_rate'])
    avg_max_heart_rate_perct = float(row['avg_max_heart_rate_perct'])
    enhanced_duration = datetime.strptime(str(row['enhanced_duration']), '%H:%M:%S')
    enhanced_duration = enhanced_duration.time()
    weekend = bool(row['weekend'])
    day_of_week = int(row['day_of_week'])
    hour = int(row['hour'])
    day_cat = row['day_cat']
    xx += row['distance']

    df1.loc[r] = [userID,
                  date_AEST,
                  distance,
                  enhanced_avg_speed,
                  avg_power,
                  avg_wpkg,
                  avg_heart_rate,
                  avg_max_heart_rate_perct,
                  enhanced_duration,
                  weekend,
                  day_of_week,
                  hour,
                  day_cat]
    r += 1

df1.to_csv("data.csv")
df2 = pd.read_csv("data.csv")
#print(df2[["day_of_week", "distance"]])
print(df2.head())
##################


sql_query = """
    SELECT
        *
    FROM
      `redbackoperationsdataai.fitness_user_summary_features.feature-summary-filtered-user1`
    WHERE avg_power IS NOT NULL AND enhanced_avg_speed IS NOT NULL
    --LIMIT 100
"""

job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
results = client.query(sql_query, job_config=job_config)
xx = 0

df3 = pd.DataFrame(columns=['userID',
                            'date_AEST',
                            'distance',
                            'enhanced_avg_speed',
                            'avg_power',
                            'avg_wpkg',
                            'avg_heart_rate',
                            'avg_max_heart_rate_perct',
                            'enhanced_duration',
                            'weekend',
                            'day_of_week',
                            'hour',
                            'day_cat'])

r = 0;
for row in results:
    userID = row['userID']
    date_AEST = datetime.strptime(str(row['date_AEST']), '%Y-%m-%d')
    date_AEST = date_AEST.date()
    distance = float((row['distance']))
    enhanced_avg_speed = float(row['enhanced_avg_speed'])
    avg_power = float(row['avg_power'])
    avg_wpkg = float(row['avg_wpkg'])
    avg_heart_rate = float(row['avg_heart_rate'])
    avg_max_heart_rate_perct = float(row['avg_max_heart_rate_perct'])
    enhanced_duration = datetime.strptime(str(row['enhanced_duration']), '%H:%M:%S')
    enhanced_duration = enhanced_duration.time()
    weekend = bool(row['weekend'])
    day_of_week = int(row['day_of_week'])
    hour = int(row['hour'])
    day_cat = row['day_cat']
    xx += row['distance']

    df3.loc[r] = [userID,
                  date_AEST,
                  distance,
                  enhanced_avg_speed,
                  avg_power,
                  avg_wpkg,
                  avg_heart_rate,
                  avg_max_heart_rate_perct,
                  enhanced_duration,
                  weekend,
                  day_of_week,
                  hour,
                  day_cat]
    r += 1

df3.to_csv("data_user.csv")
df4 = pd.read_csv("data_user.csv")
#print(df4[["day_of_week", "distance"]])
print(df4.head())
##################

x = df2["distance"].to_numpy()
y = df2["enhanced_avg_speed"].to_numpy()
x2 = df4["distance"].to_numpy()
y2 = df4["enhanced_avg_speed"].to_numpy()

plt.subplot(511)
plt.xlim(0, 200)
plt.ylim(0, 50)
plt.scatter(x,y)
plt.ylabel('Speed')
plt.xlabel('Distance')
plt.title("User 1 vs Sample")
plt.grid(True)
#
# plt.subplot(313)
plt.xlim(0, 200)
plt.ylim(0, 50)
plt.scatter(x2,y2)
plt.ylabel('Speed')
plt.grid(True)

plt.subplot(513)
plt.xlim(0, 200)
plt.ylim(0, 50)
plt.scatter(x,y)
plt.ylabel('Speed')
plt.xlabel('Distance')
plt.title("Samples")
plt.grid()

plt.subplot(515)
plt.xlim(0, 200)
plt.ylim(0, 50)
plt.scatter(x2,y2)
plt.ylabel('Speed')
plt.xlabel('Distance')
plt.title("User 1")
plt.grid()

plt.show()



# df2['day_of_week'].plot(kind='hist', edgecolor='black')
#
# plt.show()
#
#
# ax = df2.plot(
#    x='distance',
#    y='enhanced_avg_speed',
#    kind='scatter',
# )
# df4.plot(ax=ax, x='distance', y='enhanced_avg_speed', kind='scatter')
# plt.show()

# df4.plot.subplot(
#    x='distance',
#    y='enhanced_avg_speed',
#    kind='scatter'
# )
# df2.plot.subplot(
#    x='distance',
#    y='enhanced_avg_speed',
#    kind='scatter'
# )

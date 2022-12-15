
from google.cloud import bigquery
import os
from google.oauth2 import service_account
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np

# --------------------
# M.TELLEY
# --------------------

runanalysis = True
if runanalysis:
    # IF csv exists - Ask user if they want to connect to bigquery to run script or use existing csv file
    # ELSE - no csv file, connect to bigquery and run script.
    # Note: A basic CSV is provided.
    file_exists_1 = os.path.exists('data.csv')
    file_exists_2 = os.path.exists('redbackoperationsdataai-f939ef3426ea.json')
    runScript = 0
    if file_exists_1 and file_exists_2:
        userAns = int(input(
            "Local data is available and Bigquery auth available!\nWould you like to run the SQL scripts? 1 - Yes, 0 - No: "))
        if userAns == 1:
            # Set up connection with BigQuery Project tp retrive data
            credentials = service_account.Credentials.from_service_account_file(
                'redbackoperationsdataai-f939ef3426ea.json')
            project_id = 'redbackoperationsdataai'
            client = bigquery.Client(credentials=credentials, project=project_id)
            print("Client creating using", project_id, "project: {}".format(client.project))

            # --------------------
            # Get Data from Database
            # --------------------
            # Get Data
            print("Attempting to run SQL Script #1")
            sql_query = """
            SELECT
              *
            FROM
              `redbackoperationsdataai.Master_Fitness_Data.master-fitness-minute-summary`
            WHERE
              date = '2022-11-19'
              OR date = '2022-11-17')
            ORDER BY
              date,
              hour,
              minute
            """
            job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
            results = client.query(sql_query, job_config=job_config)

            # --------------------
            # Organise SQL results/data into dataframe
            # --------------------
            df1 = pd.DataFrame(columns=['indexvalue',
                                        'userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'distance_per_minute',
                                        'enhanced_speed',
                                        'power',
                                        'wpkg',
                                        'FTP_perct',
                                        'cadence',
                                        'grade',
                                        'enhanced_altitude',
                                        'hear_rate',
                                        'max_heart_rate_perct'])
            r = 0;
            for row in results:
                indexvalue = r
                userID = row['userID']
                date = row['date']
                Hour = row['Hour']
                Minute = row['Minute']
                distance_per_minute = row['distance_per_minute']
                enhanced_speed = row['enhanced_speed']
                power = row['power']
                wpkg = row['wpkg']
                FTP_perct = row['FTP_perct']
                cadence = row['cadence']
                grade = row['grade']
                enhanced_altitude = row['enhanced_altitude']
                hear_rate = row['hear_rate']
                max_heart_rate_perct = row['max_heart_rate_perct']

                df1.loc[r] = [indexvalue,
                              userID,
                              date,
                              Hour,
                              Minute,
                              distance_per_minute,
                              enhanced_speed,
                              power,
                              wpkg,
                              FTP_perct,
                              cadence,
                              grade,
                              enhanced_altitude,
                              hear_rate,
                              max_heart_rate_perct]
                r += 1

            # --------------------
            # Write data to local
            # --------------------
            if len(df1) == 0:
                print("Dataframe is empty!")
            else:
                print("Dataframe row count = ", len(df1))
                df1.to_csv("data.csv")
                print("Data written to csv file")

            # --------------------
            # Run second script - filtered values
            # --------------------

            print("Attempting to run SQL Script #2")
            sql_query = """
            SELECT
              userID, date, Hour, Minute, wpkg, grade
            FROM
              `redbackoperationsdataai.Master_Fitness_Data.master-fitness-minute-summary`
            WHERE
              grade != 0
              AND (date = '2022-11-19'
              OR date = '2022-11-17')
            ORDER BY
              date,
              hour,
              minute
            """
            job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
            results = client.query(sql_query, job_config=job_config)

            # --------------------
            # Organise SQL results/data into dataframe
            # --------------------
            df2 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'wpkg',
                                        'grade'])
            r = 0
            for row in results:
                userID = row['userID']
                date = row['date']
                Hour = row['Hour']
                Minute = row['Minute']
                wpkg = row['wpkg']
                grade = row['grade']

                df2.loc[r] = [userID,
                              date,
                              Hour,
                              Minute,
                              wpkg,
                              grade]
                r += 1

            # --------------------
            # Write data to local
            # --------------------
            if len(df2) == 0:
                print("Dataframe is empty!")
            else:
                print("Dataframe row count = ", len(df2))
                df2.to_csv("data2.csv")
                print("Data written to csv file")

            # --------------------
            # Run third script - filtered values
            # --------------------

            print("Attempting to run SQL Script #3")
            sql_query = """
            SELECT
              userID, date, Hour, Minute, wpkg as wpkg_3, enhanced_speed as enhanced_speed_3
            FROM
              `redbackoperationsdataai.Master_Fitness_Data.master-fitness-minute-summary`
            WHERE
              grade >= 2
              AND (date = '2022-11-19'
              OR date = '2022-11-17')
            ORDER BY
              date,
              hour,
              minute
            """
            job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
            results = client.query(sql_query, job_config=job_config)

            # --------------------
            # Organise SQL results/data into dataframe
            # --------------------
            df3 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'wpkg_3',
                                        'enhanced_speed_3'])
            r = 0
            for row in results:
                userID = row['userID']
                date = row['date']
                Hour = row['Hour']
                Minute = row['Minute']
                wpkg = row['wpkg_3']
                enhanced_speed = row['enhanced_speed_3']

                df3.loc[r] = [userID,
                              date,
                              Hour,
                              Minute,
                              wpkg,
                              enhanced_speed]
                r += 1

            # --------------------
            # Write data to local
            # --------------------
            if len(df3) == 0:
                print("Dataframe is empty!")
            else:
                print("Dataframe row count = ", len(df3))
                df3.to_csv("data3.csv")
                print("Data written to csv file")
                # print(df3.head())

            # --------------------
            # Run forth script - filtered values
            # --------------------

            print("Attempting to run SQL Script #4")
            sql_query = """
            SELECT
              userID, date, Hour, Minute, cadence as cadence_4, enhanced_speed as enhanced_speed_4
            FROM
              `redbackoperationsdataai.Master_Fitness_Data.master-fitness-minute-summary`
            WHERE
              cadence >= 40
              AND (date = '2022-11-19'
              OR date = '2022-11-17')
            ORDER BY
              date,
              hour,
              minute
            """
            job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
            results = client.query(sql_query, job_config=job_config)

            # --------------------
            # Organise SQL results/data into dataframe
            # --------------------
            df4 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'cadence_4',
                                        'enhanced_speed_4'])
            r = 0
            for row in results:
                userID = row['userID']
                date = row['date']
                Hour = row['Hour']
                Minute = row['Minute']
                cadence = row['cadence_4']
                enhanced_speed = row['enhanced_speed_4']

                df4.loc[r] = [userID,
                              date,
                              Hour,
                              Minute,
                              cadence,
                              enhanced_speed]
                r += 1

            # --------------------
            # Write data to local
            # --------------------
            if len(df4) == 0:
                print("Dataframe is empty!")
            else:
                print("Dataframe row count = ", len(df4))
                df4.to_csv("data4.csv")
                print("Data written to csv file")
                # print(df3.head())
        else:
            print("Converting CSV files into Pandas Dataframes")
            df1 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'distance_per_minute',
                                        'enhanced_speed',
                                        'power',
                                        'wpkg',
                                        'FTP_perct',
                                        'cadence',
                                        'grade',
                                        'enhanced_altitude',
                                        'hear_rate',
                                        'max_heart_rate_perct'])
            df1 = pd.read_csv("data.csv")
            df2 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'wpkg',
                                        'grade'])
            df2 = pd.read_csv("data2.csv")
            df3 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'wpkg_3',
                                        'enhanced_speed_3'])
            df3 = pd.read_csv("data3.csv")
            df4 = pd.DataFrame(columns=['userID',
                                        'date',
                                        'Hour',
                                        'Minute',
                                        'cadence_4',
                                        'enhanced_speed_4'])
            df4 = pd.read_csv("data4.csv")
    else:
        print("Bigquery Auth not available - Converting CSV files into Pandas Dataframes")
        df1 = pd.DataFrame(columns=['userID',
                                    'date',
                                    'Hour',
                                    'Minute',
                                    'distance_per_minute',
                                    'enhanced_speed',
                                    'power',
                                    'wpkg',
                                    'FTP_perct',
                                    'cadence',
                                    'grade',
                                    'enhanced_altitude',
                                    'hear_rate',
                                    'max_heart_rate_perct'])
        df1 = pd.read_csv("data.csv")
        df2 = pd.DataFrame(columns=['userID',
                                    'date',
                                    'Hour',
                                    'Minute',
                                    'wpkg',
                                    'grade'])
        df2 = pd.read_csv("data2.csv")
        df3 = pd.DataFrame(columns=['userID',
                                    'date',
                                    'Hour',
                                    'Minute',
                                    'wpkg_3',
                                    'enhanced_speed_3'])
        df3 = pd.read_csv("data3.csv")
        df4 = pd.DataFrame(columns=['userID',
                                    'date',
                                    'Hour',
                                    'Minute',
                                    'cadence_4',
                                    'enhanced_speed_4'])
        df4 = pd.read_csv("data4.csv")

    # --------------------
    # Correlate Dataframe and present
    # --------------------
    # Main data table only
    dfCor = df1[['distance_per_minute',
                 'enhanced_speed',
                 'power',
                 'wpkg',
                 'FTP_perct',
                 'cadence',
                 'grade',
                 'enhanced_altitude',
                 'hear_rate',
                 'max_heart_rate_perct']]

    matrix = dfCor.corr()
    print("Correlations:")
    print(matrix)
    matrix.to_csv("data_corr.csv")

    # Present Findings:
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sn.heatmap(matrix, annot=True, cbar=True, linewidths=.3)
    # plt.show()

    # --------------------
    # Explore Correlations
    # --------------------
    # 1 - Speed/Wpkg where grade > 2
    # 2 - Cadence/Wkpg
    # 3 - Grade/Wpkg where .. look at grade != 0
    # 4 - Cadence/Speed
    # 5 - HR/Wkpg
    # 6 - HR/Cadence

    # Outliers - Check impact to Correlation performance
    cols = ["enhanced_speed", "wpkg"] # one or m!=ore
    Q1 = df1[cols].quantile(0.40)
    Q3 = df1[cols].quantile(0.50)
    IQR = Q3 - Q1

    dfc = df1[~((df1[cols] < (Q1 - 1.5 * IQR)) |(df1[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    matrix = df1[~((df1[cols] < (Q1 - 1.5 * IQR)) |(df1[cols] > (Q3 + 1.5 * IQR))).any(axis=1)].corr()
    print(matrix)
    matrix.to_csv("data_corr_2.csv")
    speed_att = dfc["enhanced_speed"].to_numpy()
    wpkg_att = dfc["wpkg"].to_numpy()

    runplots = True

    if runplots:
        print("Begin Plots")
        # GRAPH/PLOT VALUES
        # Standard
        speed_att = df1["enhanced_speed"].to_numpy()
        wpkg_att = df1["wpkg"].to_numpy()
        cadence_att = df1["cadence"].to_numpy()
        heart_rate_att = df1["max_heart_rate_perct"].to_numpy()
        FTP_perct_att = df1["FTP_perct"].to_numpy()

        # filter by cadence > 40
        speed_att_4 = df4["enhanced_speed_4"].to_numpy()
        cadence_att_4 = df4["cadence_4"].to_numpy()

        # filter by grade != 0
        grade_att_2 = df2["grade"].to_numpy()
        wpkg_att_2 = df2["wpkg"].to_numpy()

        # filtered by grade >= 2
        speed_att_3 = df3["enhanced_speed_3"].to_numpy()
        wpkg_att_3 = df3["wpkg_3"].to_numpy()

        # Speed vs wpkg where grade >=2
        plt.subplot(321)
        plt.title("Speed vs wpkg where grade >=2")
        plt.scatter(wpkg_att_3, speed_att_3, marker=".")
        trend = np.polyfit(wpkg_att_3, speed_att_3, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(wpkg_att_3, trendpoly(wpkg_att_3))
        plt.ylabel('Speed')
        plt.xlabel('Wkpg')
        plt.grid(True)

        # Wpkg vs cadence
        plt.subplot(322)
        plt.title("Wpkg vs cadence")
        plt.scatter(wpkg_att, cadence_att, marker=".")
        trend = np.polyfit(wpkg_att, cadence_att, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(wpkg_att, trendpoly(wpkg_att))
        plt.ylabel('Cadence')
        plt.xlabel('Wkpg')
        plt.grid(True)

        # Speed vs cadence
        plt.subplot(323)
        plt.title("Speed vs cadence > 40")
        plt.scatter(speed_att_4, cadence_att_4, marker=".")
        trend = np.polyfit(speed_att_4, cadence_att_4, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(speed_att_4, trendpoly(speed_att_4))
        plt.ylabel('Cadence')
        plt.xlabel('Speed')
        plt.grid(True)

        # wpkg vs HR
        plt.subplot(324)
        plt.title("wpkg vs HR")
        # plt.xlim(0, 7)
        # plt.ylim(0, 70)
        plt.scatter(FTP_perct_att, heart_rate_att, marker=".")
        trend = np.polyfit(FTP_perct_att, heart_rate_att, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(FTP_perct_att, trendpoly(FTP_perct_att))
        plt.ylabel('HR')
        plt.xlabel('FTP %')
        plt.grid(True)

        # Cadence vs HR
        plt.subplot(325)
        plt.title("Cadence vs HR")
        # plt.xlim(0, 7)
        # plt.ylim(0, 70)
        plt.scatter(cadence_att, heart_rate_att, marker=".")
        trend = np.polyfit(cadence_att, heart_rate_att, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(cadence_att, trendpoly(cadence_att))
        plt.ylabel('HR')
        plt.xlabel('Cadence')
        plt.grid(True)

        # Wpkg vs grade where grade != 0
        plt.subplot(326)
        plt.title("Wpkg vs grade where grade != 0")
        # plt.xlim(0, 7)
        # plt.ylim(0, 70)
        plt.scatter(wpkg_att_2, grade_att_2, marker=".")
        trend = np.polyfit(wpkg_att_2, grade_att_2, 1)
        trendpoly = np.poly1d(trend)
        plt.plot(wpkg_att_2, trendpoly(wpkg_att_2))
        plt.ylabel('Grade')
        plt.xlabel('Wpkg')
        plt.grid(True)

        # adjust
        plt.subplots_adjust(hspace=0.558)
        # show plot
        plt.show()

# FINAL DATASET for modelling:

# --------------------
# Get Data from Database
# --------------------
# Get Data

file_exists_m = os.path.exists('data_modelling.csv')
#runScript = input("Would you like to rerun the modelling script? Yes - 1, No - 0: ")
runScript = 0
if runScript == 1:
    credentials = service_account.Credentials.from_service_account_file('redbackoperationsdataai-f939ef3426ea.json')
    project_id = 'redbackoperationsdataai'
    client = bigquery.Client(credentials=credentials, project=project_id)
    print("Attempting to run SQL Script for modelling data")
    # GETTING USER 1 DATA FOR THE MONTH OF NOVEMBER ONLY, EST 11 SESSIONS of data (17 hours of data)
    sql_query = """
    SELECT
      userID,
      sessionID,
      step,
      enhanced_speed,
      power,
      wpkg,
      FTP_perct,
      cadence,
      grade,
      max_heart_rate_perct
    FROM
      `redbackoperationsdataai.Master_Fitness_Data.master-fitness-model-dataset`
    WHERE
      # userID = "U1000000"
       extract(YEAR from date) = 2022
      AND extract(MONTH from date) >= 10
    ORDER BY
      userID,
      date,
      hour,
      minute
    """
    # --------------------
    # Organise SQL results/data into dataframe
    # --------------------
    job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
    results = client.query(sql_query, job_config=job_config)
    df = pd.DataFrame(columns=['indexnum',
                               'userID',
                               'sessionID',
                               'step',
                               'enhanced_speed',
                               'power',
                               'wpkg',
                               'FTP_perct',
                               'cadence',
                               'grade',
                               'max_heart_rate_perct'])
    i = 0
    for row in results:
        indexnum = i
        userID = row['userID']
        sessionID = row['sessionID']
        step = float(row['step'])
        enhanced_speed = row['enhanced_speed']
        power = row['power']
        wpkg = row['wpkg']
        FTP_perct = row['FTP_perct']
        cadence = row['cadence']
        grade = row['grade']
        max_heart_rate_perct = row['max_heart_rate_perct']

        df.loc[i] = [indexnum,
                     userID,
                     sessionID,
                     step,
                     enhanced_speed,
                     power,
                     wpkg,
                     FTP_perct,
                     cadence,
                     grade,
                     max_heart_rate_perct]
        i += 1

    # --------------------
    # Write data to local
    # --------------------
    if len(df) == 0:
        print("Dataframe is empty!")
    else:
        print("Dataframe row count = ", len(df))
        df.to_csv("data_modelling.csv")
        print("Data written to csv file")
else:
    print("Converting CSV files into Pandas Dataframes")
    df = pd.DataFrame(columns=['indexnum',
                               'userID',
                               'sessionID',
                               'step',
                               'enhanced_speed',
                               'power',
                               'wpkg',
                               'FTP_perct',
                               'cadence',
                               'grade',
                               'max_heart_rate_perct'])
    df = pd.read_csv("data_modelling.csv")

# --------------------
# Visualise Data set prior to training
# --------------------

# GET SESSION IDS
sessions = df["sessionID"].unique()

# prep using s_131 as an example only:
dfr = df[df['sessionID'] == "s_131"]
step_step = dfr["step"].to_numpy().tolist()
power_step = dfr["power"].to_numpy().tolist()
dfr.to_csv("data_modelling_single_session.csv")
plt.plot(step_step,power_step)
plt.title("s_131 power time series")
plt.show()
plt.clf()
plt.cla()
plt.close()

# Data Preprocessing for Classification
del df['userID']
del df['sessionID']

# get info and check null values
print(df.info())

# T transposes the table
print(df.describe().T)

print("End of Program")

# WAHOO API / FIT HANDLING
The Wahoo API is a RESTful API that allows developers to interact with Wahoo products and services. The API is used to authenticate with Wahoo, retrieve user details, work out details, and a FIT file.

# Contents

- [WAHOO API / FIT HANDLING](#wahoo-api--fit-handling)
- [Contents](#contents)
- [](#)
  - [Authentication](#authentication)
  - [User Details](#user-details)
  - [Select Workout](#select-workout)
  - [Workout Summary / Fit File](#workout-summary--fit-file)
    - [FIT Conversion and Handling](#fit-conversion-and-handling)
  - [Data Warehouse / Table](#data-warehouse--table)
- [Other Items](#other-items)
  - [API Endpoints](#api-endpoints)
  - [Request Use of the Cloud API](#request-use-of-the-cloud-api)
  - [Client Secret Security](#client-secret-security)
  - [Token Refresh](#token-refresh)
  - [Database Handling](#database-handling)

# 

## Authentication
To authenticate with the Wahoo API, you need to provide your client ID and client secret. You can find your client ID and client secret in the Wahoo Developer Portal.

Once you have your client ID and client secret, you can use them to authenticate with the API using the following steps:

1. Mount Google Drive
2. Load API credentials from secure location
3. Get Key Cred details

The following code shows how to mount Google Drive and load the credentials from a secure location:

````{python}
import os
from google.colab import drive

## Mount Google Drive
drive.mount('/content/drive')

## Load credentials from secure location
creds_file_path = '/content/drive/MyDrive/Colab Notebooks/Assets/Wahoo/cred.txt'

## Get Key Cred details
with open(creds_file_path, 'r') as f:
    lines = f.readlines()
    _, client_id = lines[0].strip().split(" = ", 1)
    _, client_secret = lines[1].strip().split(" = ", 1)
````

Once you have your client ID and client secret, you can use them to authenticate with the API using the following code:

````{python} 
import requests

# Set OAuth2 credentials and other parameters
redirect_uri = 'https://www.heyrcg.com' 
scopes = 'user_read%20+workouts_read%20+power_zones_read%20+power_zones_write'
base_url = 'https://api.wahooligan.com'

# Step 1: Redirect user to Wahoo login page for authorisation (Sandbox)
auth_url = f'{base_url}/oauth/authorize?client_id={client_id}&redirect_uri=\
{redirect_uri}&scope={scopes}&response_type=code'
print(f'Please go to this URL to authorize the app: {auth_url}')

# User to open link, authorise and enter code
auth_code = input('Enter the code from the redirect URI: ')

# Step 2: Exchange authorisation code for access and refresh tokens
token_url = f'{base_url}/oauth/token'
payload = {
    'client_id': client_id,
    'client_secret': client_secret,
    'redirect_uri': redirect_uri,
    'grant_type': 'authorization_code',
    'code': auth_code
}

# Check Expiry Time
import time
response = requests.post(token_url, data=payload)
if response.status_code != 200:
    raise Exception(f'Error getting access token: {response.text}')
access_token = response.json()['access_token']
refresh_token = response.json()['refresh_token']
expires_in = response.json()['expires_in']
expiration_time = time.time() + expires_in
human_readable_time = time.ctime(expiration_time)
print(human_readable_time)
````

## User Details
To retrieve user details, you need to use the following steps:

1. Get User URL
2. Set headers
3. Get response
4. Print User Details

The following code shows how to get user details:

````{python}
import requests

# Get User URL
user_url = f'{base_url}/v1/user'

# Set headers
headers = {'Authorization': f'Bearer {access_token}'}

# Get response
response = requests.get(user_url, headers=headers)

# Print User Details
if response.status_code == 200:
    user_details = response.json()
    print(user_details)
else:
    print(f'Error getting user details: {response.text}')
````

## Select Workout
To select a specific workout, you need to use the following steps
1. Get Workouts URL
2. Set headers
3. Get response
4. Print list of workouts *Currently a list of 4 workouts is returned*
5. Initialise a `workoutID` variable i.e, `workoutID = workout["id"]`

````{Python}
# Get Workouts Url
workouts_url = f'{base_url}/v1/workouts?per_page=10'

# Set Headers
headers = {'Authorization': f'Bearer {access_token}'}

# Get Response 
response = requests.get(workouts_url, headers=headers)

# Print Workouts
if response.status_code != 200:
    raise Exception(f'Error getting workouts: {response.text}')
workouts = response.json()['workouts']

for i, workout in enumerate(workouts):
    print(f'{i + 1}: Workout ID {workout["id"]} on {workout["updated_at"]}')

# Select target workout (Get workoutID)
while True:
    selection = input('Select a workout (1-10): ')
    try:
        selection = int(selection)
        if 1 <= selection <= 10:
            break
        else:
            print('Invalid selection.')
    except ValueError:
        print('Invalid selection.')
````

<!-- ## Power Zone
To retrieve power zone, you need to use the following steps:

1. Get Power Zone URL
2. Set headers
3. Get response

*Note: Power Zones are not required for the fit file conversion; this was included as a discovery point to further understand how Wahoo's Cloud API works*

The following code shows how to get a power zone:

````{python}
import requests

# Get Power Zone URL
power_zone_url = f'{base_url}/v1/power_zones'

# Set headers
headers = {'Authorization': f'Bearer {access_token}'}

# Get response
response = requests.get(power_zone_url, headers=headers)

# Print Power Zone
if response.status_code == 200:
    power_zones = response.json()
    print(power_zones)
else:
    print(f'Error getting power zones: {response.text}')
```` -->


## Workout Summary / Fit File
To retrieve workout summary information, you need to use the following steps
1. Ensure a workout summary id is declared/selected (previous section)
2. Get Workout Summary URL 
3. Set headers
6. Get response
7. Get FIT File URL
8. Download and Save FIT File

The following code shows how to get FIT file:

````{python}
import requests

# Get Workout Summary URL
workout_id = workouts[selection - 1]['id'] 
workout_summary_url = f'{base_url}/v1/workouts/{workout_id}/workout_summary'

# Set Headers
headers = {'Authorization': f'Bearer {access_token}'}

# Get Response
response = requests.get(workout_summary_url, headers=headers)

# Print workout summary 
if response.status_code != 200:
    raise Exception(f'Error getting workout summary: {response.text}')
workout_summary = response.json()

# Get FIT file URL from workout summary
fit_file_url = workout_summary['file']['url']

# Download the FIT file and save it to the specified path
response = requests.get(fit_file_url)
if response.status_code != 200:
    raise Exception(f'Error downloading FIT file: {response.text}')

fit_file_path = os.path.join(folder_path, f'{workout_id}.fit')

with open(fit_file_path, 'wb') as f:
    f.write(response.content)

print(f'Downloaded FIT file for workout ID {workout_id} to {fit_file_path}')
````

### FIT Conversion and Handling
To handle and process the FIT File, The script defines two functions: `write_fit_file_to_csv` and `handlefitfile`.

`write_fit_file_to_csv` takes a `FitFile` object as input and converts its data into a CSV file format. The function filters out only the required fields from the FitFile messages and then writes them into a new CSV file. The output file path is "test_output.csv" by default, but can be changed by providing a new path as a parameter.

`handlefitfile` processes the most recent ".fit" file found in a specified directory by converting it to a ".csv" file format using the "fitparse" and "csv" libraries. It first gets the list of all files in the specified directory, filters out non-FIT files, and then gets the most recent FIT file by checking the last modified time of each file. It then converts this FIT file to a CSV file by calling the "write_fit_file_to_csv" function and saves the output CSV file into a Pandas DataFrame. The function returns the path to the output CSV file and the corresponding Pandas DataFrame.

Here are the steps involved in the `handlefitfile` function:

1. Get the list of files in the "Exports" directory.
2. Filter out non-FIT files.
3. Get the most recent FIT file in the directory.
4. Get the path of the most recent FIT file.
5. Process the file.
6. Store Data in CSV/Pandas DataFrame
7. The script saves the information to a csv file.

## Data Warehouse / Table
To ingest the FIT file into a Data Warehouse / Table, you need to use the following steps:

1. Connect to Data Warehouse
2. Create a Table
3. Insert Data
4. Close Connection

The following code provides an **EXAMPLE ONLY** of how to connect to a Data Warehouse, create a table, insert data, and close the connection:

````{Python}

# Connect to Data Warehouse
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=mydb;UID=sa;PWD=mypassword')

# Create Table
cursor = conn.cursor()
cursor.execute('CREATE TABLE fit_data (id int, activity_type varchar(255), start_time datetime, duration int, distance float, power float)')

# Insert Data
for activity in activities:
    cursor.execute('INSERT INTO fit_data (id, activity_type, start_time, duration, distance, power) VALUES (?, ?, ?, ?, ?, ?)', activity.id, activity.type, activity.start_time, activity.duration, activity.distance, activity.power)

# Close Connection
conn.close()
````
This code will connect to a SQL Server database, create a table called fit_data, and insert the data from the FIT file into the table.

# Other Items

## API Endpoints

The Wahoo API has several endpoints that can be used to interact with the Wahoo system. Below is a brief description of the most relevant ones:

- `/oauth/authorize`: This endpoint is used to authorize the client application and redirect the user to the login page. The client_id, redirect_uri, scope, and response_type are the required parameters.
- `/oauth/token`: This endpoint is used to exchange the authorization code for access and refresh tokens.
- `/v1/user`: This endpoint returns the authenticated user's details.
- `/v1/workouts`: This endpoint retrieves a list of the authenticated user's workouts.
- `/v1/workouts/{workout_id}/workout_summary`: This endpoint provides the summary of a specific workout.

## Request Use of the Cloud API

The Cloud API uses the public Wahoo server and authorised user data. Because of this, Wahoo Fitness is currently limiting the use of the API to those who request it, as well as providing more information about the scopes involved and the purpose of the application.

When you submit an application request to Wahoo, it will show up on your Developer Portal as pending approval. Be sure to include as much information as you can about your application so Wahoo can be confident in approving your use of the Cloud API.

*Note: a `Sandbox` application request was made and granted - It took around a week to get approved. I explained why I needed access; as part of Deakin's [Capstone Program](https://https://www.deakin.edu.au/information-technology/student-capstone) / [Unit SIT378](https://www.deakin.edu.au/courses/unit?unit=SIT378<br>
As an additional work item - DevOps etc should submit an application for a `Production` app/access to the Cloud API*.

## Client Secret Security

Client secrets are sensitive information and should be handled with care. They should not be embedded in the code or exposed in logs. Always store them in a secure location and use them through secure methods.

## Token Refresh

Access tokens have a limited lifespan. Once the access token expires, you can use the refresh token to get a new access token without having to go through the entire authorization process again. It is important to handle this refresh mechanism properly to provide a smooth user experience and ensure the security of user data.

## Database Handling

In the examples provided, we are connecting to a SQL Server database - This may change. Additionally, while inserting data into the database, we are not taking care of potential SQL injection issues and data type handling. This is simply an example to illustrate how you might store data. Always follow best practices when interacting with databases, such as using parameterized queries to prevent SQL injection and taking care of data types.
<br><br>

*Author: Mark Telley, 2022*

# Wahoo - Get Fit File
This script retrieves the latest workout data from the Wahoo Fitness API for a given user, saves the workout data as a .fit file in the fit_file folder, converts the .fit file to a CSV file and pandas dataframe (raw), and then processes the data for injection into a BigQuery table.

## Requirements
Python 3.6 or higher
Wahoo Fitness API access credentials
BigQuery API access credentials
Required Packages:
requests
pandas
fitparse
google-cloud-bigquery

## Installation
1. Clone this repository to your local machine
2. Install the required dependencies using pip:

## Requirements
- Python 3.6 or higher
- Wahoo Fitness API access credentials
- BigQuery API access credentials

### Required Packages:
- requests
- pandas
- fitparse
- google-cloud-bigquery

## Installation
1. Clone this repository to your local machine
2. Install the required dependencies using pip:

```bash
pip install requests pandas fitparse google-cloud-bigquery
```

## Usage
To run the script, run the following command in your terminal:

```bash
python main.py
```

## Functions
### get_user_id(application_token, secret_key, username, password) -> str: 
This function retrieves the user ID of the authenticated user.
### get_latest_fit_file(application_token, secret_key, user_id) -> bytes: 
This function retrieves the latest workout data from the Wahoo Fitness API for a given user and returns the .fit file as bytes.
### save_fit_file(fit_file_content: bytes, folder_name: str) -> None: 
This function saves the .fit file as a file in the fit_file folder.
### fit_conversion() -> pandas.DataFrame: 
This function reads in the .fit file from the fit_file folder, converts it to a CSV file, and returns the data as a pandas dataframe (raw).
### fit_process_data(raw: pandas.DataFrame) -> pandas.DataFrame: 
This function processes the raw data from the fit_file folder and prepares it for injection into a BigQuery table.

## Useful Links
### Products
[Wahoo Fitness Kickr](https://au.wahoofitness.com/devices/indoor-cycling/bike-trainers/kickr-buy)
[Wahoo Fitness Elemnt Bolt](https://au.wahoofitness.com/devices/bike-computers/elemnt-bolt-buy)

### Docs
[Wahoo Fitness Developers](https://developers.wahooligan.com/)
[Wahoo Fitness API Documentation](https://cloud-api.wahooligan.com/#introduction)
[Wahoo Fitness Workout Summary Documentation](https://cloud-api.wahooligan.com/#workout-summary)
[Google Cloud BigQuery Python API Reference](https://cloud.google.com/python/docs/reference/bigquery/latest)
[Garmin FIT SDK Overview](https://developer.garmin.com/fit/overview/)
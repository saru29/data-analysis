import os
import requests

# GETS USER ID
def get_user_id(application_token, secret_key, username, password):
    # Get the access token
    auth_response = requests.post(
        'https://api.wahooligan.com/v1/oauth2/token',
        data={
            'grant_type': 'password',
            'client_id': application_token,
            'client_secret': secret_key,
            'username': username,
            'password': password
        }
    )
    access_token = auth_response.json()['access_token']

    # Get the authenticated user's ID
    user_response = requests.get(
        'https://api.wahooligan.com/v1/user',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    user_id = user_response.json()['id']

    return user_id

# GETS LATEST (LAST) FIT FILE FOR USER
def get_latest_fit_file(application_token, secret_key, user_id):
    # Get the access token
    auth_response = requests.post(
        'https://api.wahooligan.com/v1/oauth2/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': application_token,
            'client_secret': secret_key
        }
    )
    access_token = auth_response.json()['access_token']

    # Get the user's workouts
    workouts_response = requests.get(
        f'https://api.wahooligan.com/v1/users/{user_id}/workouts',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    workouts = workouts_response.json()

    # Get the workout summary for the latest workout
    latest_workout = workouts[0]
    workout_id = latest_workout['id']
    summary_response = requests.get(
        f'https://api.wahooligan.com/v1/workouts/{workout_id}/workout_summary',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    workout_summary = summary_response.json()

    # Get the URL of the latest .fit file
    latest_fit_url = workout_summary['file']['url']

    # Download the .fit file
    fit_response = requests.get(latest_fit_url)
    return fit_response.content


# SAVE FIT FILE TO FOLDER LOCATION
def save_fit_file(fit_file_content, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    fit_file_path = folder_name + "/latest_workout.fit"

    with open(fit_file_path, "wb") as f:
        f.write(fit_file_content)

    print("FIT file saved as " + fit_file_path)


# RUN
def run_get_save_fit_file(application_token, secret_key, user_id):
    latest_fit_content = get_latest_fit_file(application_token, secret_key, user_id)
    save_fit_file(latest_fit_content, "fit_file")
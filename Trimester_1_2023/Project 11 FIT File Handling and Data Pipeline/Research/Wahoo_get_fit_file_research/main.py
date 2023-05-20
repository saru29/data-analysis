# Get FIT WILL FROM WAHOO / SAVE TO Bigquery
# Mark Telley - Deakin Uni.

from getfitfile import run_get_save_fit_file, get_user_id

# User Details
# TODO: Remove! prior to Github commit
username = "username"
password = "password"

# API details
# TODO: Remove! prior to Github commit
application_token = "WAHOOAPP"
secret_key = "NOTSOSECRETKEY"


if __name__ == '__main__':
    # GET USER ID
    user_id = get_user_id(application_token, secret_key, username, password)

    # GET FILE
    run_get_save_fit_file(application_token, secret_key, user_id)

    # CONVERT FIT FILE TO CSV
    # TODO: Pass FIT file to conversion script

    # PREPROCESS DATA
    # TODO: Read CSV/Pandas DATA Frame, and prep for table creation in Bigquery (Google API)


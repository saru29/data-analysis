# BLE Cycling Power Data Collection Readme

This Python script collects cycling power data from a Bluetooth Low Energy (BLE) device and saves it to a CSV file. The script uses the `asyncio` and `bleak` libraries for working with BLE devices, and the `pandas` library for data manipulation and storage.

## Requirements

- Python 3.9
- `asyncio` library
- `bleak` library
- `pandas` library

## Usage

1. Clone or download the script to your local machine.
2. Install the required libraries using pip:

    ```
    pip install asyncio bleak pandas
    ```

3. Connect to your BLE device and obtain its MAC address.
4. Replace the `device_address` variable in the script with the MAC address of your BLE device.
5. Run the script:

    ```
    python main.py
    ```

The script will run for the specified `session_length` (in seconds) and save the collected data to a CSV file named "Activity_Data.csv" in the same directory as the script.

## Data Collection

The script collects the following data points:

- `step`: The timestamp of the data point in the format `YYYY/MM/DD HH:MM:SS`.
- `instantaneous_power`: The current power output in watts.
- `accumulated_energy`: The total energy accumulated in joules.
- `pedal_power_balance`: The balance of power between the left and right pedals as a percentage.
- `accumulated_torque`: The total torque accumulated in Nm.
- `cumulative_wheel_revs`: The total number of wheel revolutions.
- `last_wheel_event_time`: The timestamp of the last wheel event in the format `YYYY/MM/DD HH:MM:SS`.
- `cumulative_crank_revs`: The total number of crank revolutions.
- `last_crank_event_time`: The timestamp of the last crank event in the format `YYYY/MM/DD HH:MM:SS`.
- `maximum_force_magnitude`: The maximum force applied to the pedals in Newtons.
- `minimum_force_magnitude`: The minimum force applied to the pedals in Newtons.
- `maximum_torque_magnitude`: The maximum torque applied to the pedals in Nm.
- `minimum_torque_magnitude`: The minimum torque applied to the pedals in Nm.
- `top_dead_spot_angle`: The position of the top dead spot of the pedal stroke in degrees.
- `bottom_dead_spot_angle`: The position of the bottom dead spot of the pedal stroke in degrees.

## Data Storage

The collected data is stored in a pandas dataframe and saved to a CSV file named "Activity_Data.csv" in the same directory as the script. The script can be modified to store the data in other formats or to upload the data to a cloud service such as Google BigQuery.

## Images

![Alt text](https://github.com/redbackoperations/data-analysis/blob/main/Trimester_1_2023/Project%2011%20FIT%20File%20Handling%20and%20Data%20Pipeline/Wahoo_Kickr_Connection/Images/IMG_6064.JPG "Example image 1")

![Alt text](https://github.com/redbackoperations/data-analysis/blob/main/Trimester_1_2023/Project%2011%20FIT%20File%20Handling%20and%20Data%20Pipeline/Wahoo_Kickr_Connection/Images/IMG_6068_1.JPG "Example image 1")

## Key Resource:

[PyCycling package](https://pypi.org/project/pycycling/#description)

import asyncio
from bleak import BleakClient
import pandas as pd
from cycling_power_service import CyclingPowerService
from datetime import datetime

# define the length of the data collection session
session_length = 30.0

# define an async function for running the data collection
async def run(address, session_length):
    # connect to the BLE device
    async with BleakClient(address) as client:
        data_list = []  # create an empty list to store the data points
        def my_measurement_handler(data):
            # format the current time as a string for the step column in the dataframe
            formatted_step = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            # add the data to the list as a dictionary
            data_list.append({
                'step': formatted_step,
                'instantaneous_power': data.instantaneous_power,
                'accumulated_energy': data.accumulated_energy,
                'pedal_power_balance': data.pedal_power_balance,
                'accumulated_torque': data.accumulated_torque,
                'cumulative_wheel_revs': data.cumulative_wheel_revs,
                'last_wheel_event_time': data.last_wheel_event_time,
                'cumulative_crank_revs': data.cumulative_crank_revs,
                'last_crank_event_time': data.last_crank_event_time,
                'maximum_force_magnitude': data.maximum_force_magnitude,
                'minimum_force_magnitude': data.minimum_force_magnitude,
                'maximum_torque_magnitude': data.maximum_torque_magnitude,
                'minimum_torque_magnitude': data.minimum_torque_magnitude,
                'top_dead_spot_angle': data.top_dead_spot_angle,
                'bottom_dead_spot_angle': data.bottom_dead_spot_angle
            })
            # print the received data to the console
            print(f'Power (w):{data.instantaneous_power}')

        # check if the client is connected
        await client.is_connected()
        # create a CyclingPowerService instance for the client
        trainer = CyclingPowerService(client)
        # set the measurement handler for the service
        trainer.set_cycling_power_measurement_handler(my_measurement_handler)
        # hear rate data HERE <><><><>
        # enable notifications for the service
        await trainer.enable_cycling_power_measurement_notifications()
        # wait for the session length to elapse
        await asyncio.sleep(session_length)
        # disable notifications for the service
        await trainer.disable_cycling_power_measurement_notifications()

    # create a pandas dataframe from the data list
    df = pd.DataFrame(data_list)
    # save the dataframe to a CSV file
    df.to_csv("Activity_Data.csv")
    # print the dataframe to the console
    print(df)

if __name__ == "__main__":
    import os
    os.environ["PYTHONASYNCIODEBUG"] = str(1)
    # TODO: REMOVE ADDRESS - the MAC address of the BLE device to connect to
    device_address = "C2:XX:08"

    if True:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run(device_address, session_length))

    # Bigquery Export here <>


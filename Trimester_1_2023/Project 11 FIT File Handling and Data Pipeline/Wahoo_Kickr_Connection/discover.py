"""
An example script which lists all available bluetooth devices. Use this to obtain the device_address used in other
scripts
"""

import asyncio
from bleak import discover


async def run():
    devices = await discover()
    for d in devices:
        print(d.name, " ", d)
        # TICKR 4CB2   FE:A9:E6:88:70:2B: TICKR 4CB2
        # KICKR CORE 6965   C2:8E:F1:F2:1A:08: KICKR CORE 6965

if __name__ == "__main__":
    import os

    os.environ["PYTHONASYNCIODEBUG"] = str(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

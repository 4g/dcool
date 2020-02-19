import time
from dc_env import DataCenterEnv
import threading

class DataCenterLife(threading.Thread):
    def __init__(self, dc):
        threading.Thread.__init__(self)
        self.dc = dc

    def run(self):
        self.event1()

    def add_random_heater(self):
        pass

    def delete_random_heater(self):
        pass

    def delete_random_fan(self):
        pass

    def add_random_fan(self):
        pass

    def event1(self):
        """
        200 seconds
        1. first heater heats up to rate 5 and stays for 20s
        2. first heater cools back to rate 3 stays for 5 seconds
        3. new heater comes into being with rate 3, blocking path to heater 1, stays for 30s
        4. both heater go down to rate 1, stays for 30s
        5. both increase to rate 5, stays for 30s
        :return: nothing
        """
        self.dc.add_heater(450, 250, rate=0)
        self.dc.add_heater(250, 200, rate=0)

        time.sleep(5)
        self.dc.heaters[0].rate = 1
        time.sleep(5)
        self.dc.heaters[0].rate = 5
        time.sleep(20)
        self.dc.heaters[0].rate = 3
        time.sleep(20)
        self.dc.heaters[1].rate = 5
        time.sleep(30)
        self.dc.heaters[0].rate = 1
        self.dc.heaters[1].rate = 1
        time.sleep(30)
        self.dc.heaters[0].rate = 5
        self.dc.heaters[1].rate = 5
        time.sleep(20)
        self.dc.heaters[0].rate = 0
        self.dc.heaters[1].rate = 0
        time.sleep(20)
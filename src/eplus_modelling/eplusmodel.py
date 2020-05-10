import opyplus as op
class EplusModelDC:
    def __init__(self, idf_file, weather_file):
        self.epm = op.Epm.from_idf(idf_file)
        self.weather = weather_file
        self.schedules = []
        self.setpoints = []
        self.setup()

    def setup(self):
        schedules = []
        setpoints = []

        for table in self.epm:
            if len(table) != 0:
                name = table.get_name()
                if "schedule" in name.lower():
                    schedules.append(table)
                if "setpoint" in name.lower():
                    setpoints.append(table)

        self.schedules = schedules
        self.setpoints = setpoints
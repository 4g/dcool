import opyplus as op

class EplusModelDC:
    def __init__(self, idf_file, weather_file):
        self.epm = op.Epm.from_idf(idf_file)
        self.weather = weather_file
        self.schedules = {}
        self.setpoints = {}
        self.schedule_index_map = {}
        self.schedule_values = {}
        self.setup()

    def setup(self):
        schedules = {}
        setpoints = {}

        for table in self.epm:
            if len(table) != 0:
                name = table.get_name()
                if "schedule" in name.lower():
                    for schedule in table:
                        schedules[schedule.name] = schedule

                if "setpoint" in name.lower():
                    for setpoint in table:
                        setpoints[name] = setpoint

        self.schedules = schedules
        self.setpoints = setpoints

        for s in self.schedules:
            self.parse_schedule(self.schedules[s])

        # delete all periods except the first
        first_period = True
        for period in self.epm.RunPeriod:
            if first_period:
                first_period = False
                continue
            period.delete()

    def set_period(self, start, end):
        period = self.epm.RunPeriod[0]
        self.modify_period(period, start, end)

    @staticmethod
    def modify_period(period, start, end):
        begin_month = 1
        begin_day_of_month = 2
        end_month = 4
        end_day_of_month = 5

        period[begin_month] = start[1]
        period[begin_day_of_month] = start[0]
        period[end_month] = end[1]
        period[end_day_of_month] = end[0]

    def isvalue(self, x):
        try:
            float(x)
            return True
        except:
            return False

    def parse_schedule(self, schedule):
        data = schedule.to_dict()
        for key, value in data.items():
            if isinstance(value, str) and self.isvalue(value):
                self.schedule_index_map[schedule.name] = self.schedule_index_map.get(schedule.name, [])
                self.schedule_index_map[schedule.name].append(key)
                self.schedule_values[schedule.name] = value

    def update_schedule(self, name, value):
        schedule_keys = self.schedule_index_map[name]
        for key in schedule_keys:
            self.schedules[name][key] = value

    def simulate(self, odir):
        return op.simulate(self.epm, self.weather, odir)

    def get_modifiables(self):
        return self.schedule_values

class EplusExperiment:
    def __init__(self, name):
        self.eplusmodel_a = EplusModelDC("eplus_modelling/dc_specs/2ZoneDataCenterHVAC_wEconomizer.idf",
                                         "eplus_modelling/dc_specs/IND_Bangalore.432950_ISHRAE.epw")
        self.name = name

    def set_a(self, name, a):
        self.eplusmodel_a.update_schedule(name, a)

    def run(self):
        sim_a = self.eplusmodel_a.simulate(self.name + "_a")
        a_eso = sim_a.get_out_eso().get_data()
        return a_eso

    def set_period(self, start, end):
        self.eplusmodel_a.set_period(start, end)

    def get_modifiables(self):
        return self.eplusmodel_a.get_modifiables()

if __name__ == "__main__":
    eplusmodel = EplusExperiment("heating_cooling_setpoints")
    print(eplusmodel.get_modifiables())
    eplusmodel.set_period(start=(1, 1), end=(31, 12))
    eplusmodel.set_a("heating setpoints", str(15))
    eplusmodel.set_a("cooling setpoints", str(18))
    a = eplusmodel.run()
    import json
    # print(json.dumps(eplusmodel.eplusmodel_a.get_modifiables()))
    for col in a.columns:
        print(col)

import os
import opyplus as op

eplus_dir_path = op.get_eplus_base_dir_path((9, 1, 0))

base_idf_path = os.path.join("dc_specs",
    "2ZoneDataCenterHVAC_wEconomizer.idf"
    )

epw_path = os.path.join(
    eplus_dir_path,
    "WeatherData",
    "USA_CO_Golden-NREL.724666_TMY3.epw"
    )

epm = op.Epm.from_idf(base_idf_path)
# print(epm)
    #
    # def extract_values(d):
    #     for elem in d:
    #

# op.simulate(epm, epw_path, base_dir_path=".", simulation_name="test1")

schedules = []
setpoints = []

for table in epm:
    if len(table) != 0:
        name = table.get_name()
        if "schedule" in name.lower():
            schedules.append(table)

        if "setpoint" in name.lower():
            setpoints.append(table)

print(schedules)
print(setpoints)

for schedule in schedules:
    for elem in schedule:
        print(elem)


# for period in epm.schedule_compact:
#     # print(period.get_info())
#     # print(period['field_1'])
#     print(period.name, period.to_dict())
#     # break
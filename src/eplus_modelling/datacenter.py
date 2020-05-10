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
print(epm)

for period in epm.schedule_compact:
    # print(period.get_info())
    print(period.name)

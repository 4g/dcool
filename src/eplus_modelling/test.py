import os
import opyplus as op

eplus_dir_path = op.get_eplus_base_dir_path((9, 1, 0))

# idf path
idf_path = os.path.join(
    eplus_dir_path,
    "ExampleFiles",
    "2ZoneDataCenterHVAC_wEconomizer.idf"
)

# epw path
epw_path = os.path.join(
    eplus_dir_path,
    "WeatherData",
    "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
)

# run simulation
s = op.simulate(idf_path, epw_path, "my-first-simulation")

print(s.get_info())

print(f"status: {s.get_status()}\n")
print(f"Eplus .err file:\n{s.get_out_err().get_content()}")

# retrieve hourly output (.eso file)
hourly_output = s.get_out_eso()

# ask for datetime index on year 2013
hourly_output.create_datetime_index(2013)

# get Pandas dataframe
df = hourly_output.get_data()

df.to_csv("test.csv")

# # monthly resample and display
# print(df[[
#     "environment,Site Outdoor Air Drybulb Temperature",
#     "main zone,Zone Mean Air Temperature"
# ]].resample("MS").mean())

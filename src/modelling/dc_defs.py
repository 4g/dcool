chiller = "CHILLER-{n} DATA"
a_humidity = "Ambient Humidity"
a_temperature = "Ambient Temperature"

# chiller_output_params = ['SECONDARY PUMP {n}/TOTAL_POWER',
#                          'CT {n}-2/TOTAL_POWER',
#                          'CT {n}-1/TOTAL_POWER',
#                          'CH_{n}_POWER',
#                          'CWP_{n}_POWER',
#                          'CONDENSER PUMP {n}/TOTAL_POWER']

# chiller_input_params = ['CH_{n}_SUPPLY',
#                         'CH_{n}_RETURN',
#                         'oat1',
#                         'oah',
#                         'CW_{n}_RETURN',
#                         'CW_{n}_SUPPLY',
#                         'CH_{n}/Evaporator Saturation Temp']

chiller_output_params = ['CH_{n}_POWER',
                         'CH_{n}_SUPPLY']

chiller_input_params = ['CH_{n}_RETURN',
                        'oat1',
                        'oah',
                        'CW_{n}_RETURN',
                        'CH_{n}/Evaporator Saturation Temp']

pue_full_input = ['PDU/ZONE_2/DC_TOTAL_LOAD', 'oat1', 'oah', 'CH_{n}/Evaporator Saturation Temp']
pue_full_output = ['PDU/ZONE_2/PUE']

pahu_input_params = ["PAHU_SA_SP", "PAHU_RA_SP", "PAHU RETURN_TEMP"]
pahu_output_params = ["PAHU FAN_SPEED", "PAHU SUP_TEMP"]

data_hall_input_params = ["POD TEMP-{n}", "POD HUM-{n}", "PAHU RA SP", "PAHU SA SP", "CH_{n}_SUPPLY", "IT LOAD"]
data_hall_output_params = ["PUE", "PAHU-{n} TOTAL POWER"]


param_dict = {
    # "pue_full" : [pue_full_input, pue_full_output],
    # "pahu": [pahu_input_params, pahu_output_params],
    # "data_hall": [data_hall_input_params, data_hall_output_params],
    "chiller": [chiller_input_params, chiller_output_params, 2]
}

# system_map = {
#     chiller.input: pahu.output1,
#     chiller.output: pahu.input1,
#     data_hall.input: pahu.output2,
#     data_hall.output: pahu.input2,
# }

data_file = "/home/apurva/work/projects/dcool/data/flipkart/timesorted_serialized_clean.csv"
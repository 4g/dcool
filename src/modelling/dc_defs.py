chiller1 = "CHILLER-1 DATA"
chiller2 = "CHILLER-2 DATA"
a_humidity = "Ambient Humidity"
a_temperature = "Ambient Temperature"

pahu_tag = ""

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

chiller_output_params = ['CH_{n}_POWER']

chiller_input_params = ['CH_{n}_RETURN',
                        'oat1',
                        'oah',
                        'CW_{n}_RETURN',
                        'CH_{n}/Evaporator Saturation Temp']

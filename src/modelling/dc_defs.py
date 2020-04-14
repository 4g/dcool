chiller = "CHILLER-{n} DATA"
a_humidity = "Ambient Humidity"
a_temperature = "Ambient Temperature"

all_params = ['timestamp', 'CH_1/Evaporator Saturation Temp', 'CH_1_POWER', 'CH_1_RETURN', 'CH_1_SUPPLY',
              'CH_2/Evaporator Saturation Temp', 'CH_2_POWER', 'CH_2_RETURN', 'CH_2_SUPPLY',
              'CONDENSER PUMP 1/TOTAL_POWER', 'CONDENSER PUMP 2/TOTAL_POWER', 'CT 1-1/TOTAL_POWER',
              'CT 1-2/TOTAL_POWER', 'CT 2-1/TOTAL_POWER', 'CT 2-2/TOTAL_POWER', 'CW_1_RETURN', 'CW_1_SUPPLY',
              'CW_2_RETURN', 'CW_2_SUPPLY', 'CWP_1_POWER', 'CWP_2_POWER', 'EM/MDB 1 PAHU/TOTAL_POWER',
              'EM/MDB 3 PAHU/TOTAL_POWER', 'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_1', 'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_2',
              'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_3', 'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_4', 'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_5',
              'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_7', 'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_1', 'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_2',
              'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_3', 'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_4', 'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_5',
              'HUMIDITY_SENSOR/Z1S2_PDU_HUMI_7', 'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_1', 'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_4',
              'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_6', 'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_2', 'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_3',
              'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_4', 'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_5', 'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_6',
              'HUMIDITY_SENSOR/Z2S2_PDU_HUMI_7', 'PDU/ZONE_2/DC_TOTAL_LOAD', 'PDU/ZONE_2/PUE',
              'SECONDARY PUMP 1/TOTAL_POWER', 'SECONDARY PUMP 2/TOTAL_POWER', 'SF/Z1 PAHU 1/FAN_SPEED',
              'SF/Z1 PAHU 1/RETURN_TEMP', 'SF/Z1 PAHU 1/SUP_TEMP', 'SF/Z1 PAHU 1/VALVE_OUT', 'SF/Z1 PAHU 2/FAN_SPEED',
              'SF/Z1 PAHU 2/RETURN_TEMP', 'SF/Z1 PAHU 2/SUP_TEMP', 'SF/Z1 PAHU 2/VALVE_OUT', 'SF/Z1 PAHU 3/FAN_SPEED',
              'SF/Z1 PAHU 3/RETURN_TEMP', 'SF/Z1 PAHU 3/SUP_TEMP', 'SF/Z1 PAHU 3/VALVE_OUT', 'SF/Z1 PAHU 4/FAN_SPEED',
              'SF/Z1 PAHU 4/RETURN_TEMP', 'SF/Z1 PAHU 4/SUP_TEMP', 'SF/Z1 PAHU 4/VALVE_OUT', 'SF/Z1 PAHU 5/FAN_SPEED',
              'SF/Z1 PAHU 5/RETURN_TEMP', 'SF/Z1 PAHU 5/SUP_TEMP', 'SF/Z1 PAHU 5/VALVE_OUT', 'SF/Z1 PAHU 6/FAN_SPEED',
              'SF/Z1 PAHU 6/RETURN_TEMP', 'SF/Z1 PAHU 6/SUP_TEMP', 'SF/Z1 PAHU 6/VALVE_OUT', 'SF/Z1 PAHU 7/FAN_SPEED',
              'SF/Z1 PAHU 7/RETURN_TEMP', 'SF/Z1 PAHU 7/SUP_TEMP', 'SF/Z1 PAHU 7/VALVE_OUT', 'SF/Z1 PAHU 8/FAN_SPEED',
              'SF/Z1 PAHU 8/RETURN_TEMP', 'SF/Z1 PAHU 8/SUP_TEMP', 'SF/Z1 PAHU 8/VALVE_OUT', 'SF/Z2 PAHU 1/FAN_SPEED',
              'SF/Z2 PAHU 1/RETURN_TEMP', 'SF/Z2 PAHU 1/SUP_TEMP', 'SF/Z2 PAHU 1/VALVE_OUT', 'SF/Z2 PAHU 2/FAN_SPEED',
              'SF/Z2 PAHU 2/RETURN_TEMP', 'SF/Z2 PAHU 2/SUP_TEMP', 'SF/Z2 PAHU 2/VALVE_OUT', 'SF/Z2 PAHU 3/FAN_SPEED',
              'SF/Z2 PAHU 3/RETURN_TEMP', 'SF/Z2 PAHU 3/SUP_TEMP', 'SF/Z2 PAHU 3/VALVE_OUT', 'SF/Z2 PAHU 4/FAN_SPEED',
              'SF/Z2 PAHU 4/RETURN_TEMP', 'SF/Z2 PAHU 4/SUP_TEMP', 'SF/Z2 PAHU 4/VALVE_OUT', 'SF/Z2 PAHU 5/FAN_SPEED',
              'SF/Z2 PAHU 5/RETURN_TEMP', 'SF/Z2 PAHU 5/SUP_TEMP', 'SF/Z2 PAHU 5/VALVE_OUT', 'SF/Z2 PAHU 6/FAN_SPEED',
              'SF/Z2 PAHU 6/RETURN_TEMP', 'SF/Z2 PAHU 6/SUP_TEMP', 'SF/Z2 PAHU 6/VALVE_OUT', 'SF/Z2 PAHU 7/FAN_SPEED',
              'SF/Z2 PAHU 7/RETURN_TEMP', 'SF/Z2 PAHU 7/SUP_TEMP', 'SF/Z2 PAHU 7/VALVE_OUT', 'SF/Z2 PAHU 8/FAN_SPEED',
              'SF/Z2 PAHU 8/RETURN_TEMP', 'SF/Z2 PAHU 8/SUP_TEMP', 'SF/Z2 PAHU 8/VALVE_OUT',
              'TEMP_SENSOR/Z1S1_PDU_TEMP_1', 'TEMP_SENSOR/Z1S1_PDU_TEMP_2', 'TEMP_SENSOR/Z1S1_PDU_TEMP_3',
              'TEMP_SENSOR/Z1S1_PDU_TEMP_4', 'TEMP_SENSOR/Z1S1_PDU_TEMP_5', 'TEMP_SENSOR/Z1S1_PDU_TEMP_6',
              'TEMP_SENSOR/Z1S1_PDU_TEMP_7', 'TEMP_SENSOR/Z1S2_PDU_TEMP_1', 'TEMP_SENSOR/Z1S2_PDU_TEMP_2',
              'TEMP_SENSOR/Z1S2_PDU_TEMP_3', 'TEMP_SENSOR/Z1S2_PDU_TEMP_4', 'TEMP_SENSOR/Z1S2_PDU_TEMP_5',
              'TEMP_SENSOR/Z1S2_PDU_TEMP_7', 'TEMP_SENSOR/Z2S1_PDU_TEMP_1', 'TEMP_SENSOR/Z2S1_PDU_TEMP_2',
              'TEMP_SENSOR/Z2S1_PDU_TEMP_3', 'TEMP_SENSOR/Z2S1_PDU_TEMP_5', 'TEMP_SENSOR/Z2S1_PDU_TEMP_6',
              'TEMP_SENSOR/Z2S1_PDU_TEMP_7', 'TEMP_SENSOR/Z2S2_PDU_TEMP_1', 'TEMP_SENSOR/Z2S2_PDU_TEMP_2',
              'TEMP_SENSOR/Z2S2_PDU_TEMP_3', 'TEMP_SENSOR/Z2S2_PDU_TEMP_4', 'TEMP_SENSOR/Z2S2_PDU_TEMP_5',
              'TEMP_SENSOR/Z2S2_PDU_TEMP_7', 'Z1-PAHU-1_RA_SP', 'Z1-PAHU-1_SA_SP', 'Z1-PAHU-2_RA_SP', 'Z1-PAHU-2_SA_SP',
              'Z1-PAHU-3_RA_SP', 'Z1-PAHU-3_SA_SP', 'Z1-PAHU-4_RA_SP', 'Z1-PAHU-4_SA_SP', 'Z1-PAHU-5_RA_SP',
              'Z1-PAHU-5_SA_SP', 'Z1-PAHU-6_RA_SP', 'Z1-PAHU-6_SA_SP', 'Z1-PAHU-7_RA_SP', 'Z1-PAHU-7_SA_SP',
              'Z1-PAHU-8_RA_SP', 'Z1-PAHU-8_SA_SP', 'Z2-PAHU-1_RA_SP', 'Z2-PAHU-1_SA_SP', 'Z2-PAHU-2_RA_SP',
              'Z2-PAHU-2_SA_SP', 'Z2-PAHU-3_RA_SP', 'Z2-PAHU-3_SA_SP', 'Z2-PAHU-4_RA_SP', 'Z2-PAHU-4_SA_SP',
              'Z2-PAHU-5_RA_SP', 'Z2-PAHU-5_SA_SP', 'Z2-PAHU-6_RA_SP', 'Z2-PAHU-6_SA_SP', 'Z2-PAHU-7_RA_SP',
              'Z2-PAHU-7_SA_SP', 'Z2-PAHU-8_RA_SP', 'Z2-PAHU-8_SA_SP', 'HUMIDITY_SENSOR/Z1S1_PDU_HUMI_6',
              'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_2', 'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_7', 'TEMP_SENSOR/Z2S1_PDU_TEMP_4',
              'oat1', 'oah', 'HUMIDITY_SENSOR/Z2S1_PDU_HUMI_5', 'SF/Z1 PAHU 8/SA_SET_POINT',
              'TEMP_SENSOR/Z1S2_PDU_TEMP_6', 'HT_EM/TOTAL_KW/TOTAL_2F']

def number_free_params(all_params):
    numeric_chars = ['m', 'n', 'k']
    wrap = lambda x: "{" + x + "}"
    unique_params = {}
    for param in all_params:
        tmp = ""
        n_digits = 0
        d_map = {}
        for c in param:
            if c.isdigit():
                char_rep = numeric_chars[n_digits]
                tmp += wrap(char_rep)
                d_map[char_rep] = c
                n_digits += 1
            else:
                tmp += c
        unique_params[tmp] = unique_params.get(tmp, {})
        for c in d_map:
            unique_params[tmp][c] = unique_params[tmp].get(c, [])
            unique_params[tmp][c].append(d_map[c])

    return unique_params

unique_params = number_free_params(all_params)

unique_params_names = ['timestamp',

                       'CH_{m}/Evaporator Saturation Temp',
                       'CH_{m}_RETURN',
                       'CH_{m}_SUPPLY',
                       'CW_{m}_RETURN',
                       'CW_{m}_SUPPLY',
                       'CWP_{m}_POWER',
                       'SECONDARY PUMP {m}/TOTAL_POWER',
                       'CONDENSER PUMP {m}/TOTAL_POWER',
                       'CT {m}-{n}/TOTAL_POWER',
                       'CH_{m}_POWER',

                       'EM/MDB {m} PAHU/TOTAL_POWER',
                       'SF/Z{m} PAHU {n}/FAN_SPEED',
                       'SF/Z{m} PAHU {n}/RETURN_TEMP',
                       'SF/Z{m} PAHU {n}/SUP_TEMP',
                       'SF/Z{m} PAHU {n}/VALVE_OUT',
                       'SF/Z{m} PAHU {n}/SA_SET_POINT',
                       'Z{m}-PAHU-{n}_RA_SP',
                       'Z{m}-PAHU-{n}_SA_SP',

                       'HUMIDITY_SENSOR/Z{m}S{n}_PDU_HUMI_{k}',
                       'PDU/ZONE_{m}/DC_TOTAL_LOAD',
                       'PDU/ZONE_{m}/PUE',
                       'TEMP_SENSOR/Z{m}S{n}_PDU_TEMP_{k}',
                       'HT_EM/TOTAL_KW/TOTAL_{m}F',

                       'oat{m}', 'oah']


def expand_param(param):
    params = []
    if param in unique_params:
        pdict = unique_params.get(param, {})
        keys = pdict.keys()
        values = list(zip(*list(pdict.values())))
        for val in values:
            pvalues = dict(zip(keys, val))
            s = param.format(**pvalues)
            params.append(s)
    else:
        params.append(param)

    return params


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

chiller_output_params = ['CH_{m}_POWER',
                         'CH_{m}_SUPPLY'
                         ]

chiller_input_params = [
                        'CH_{m}/Evaporator Saturation Temp',
                        'CH_{m}_RETURN',
                        'oat1',
                        'oah'
                        ]

# pue_full_input = ['PDU/ZONE_2/DC_TOTAL_LOAD', 'oat1', 'oah', 'CH_{n}/Evaporator Saturation Temp']
# pue_full_output = ['PDU/ZONE_2/PUE']

pahu_input_params = ["SF/Z{m} PAHU {n}/RETURN_TEMP", 'SF/Z{m} PAHU {n}/SUP_TEMP', 'CH_1_SUPPLY', 'CH_2_SUPPLY']
pahu_output_params = ['SF/Z{m} PAHU {n}/FAN_SPEED']

data_hall_inputs = ['HUMIDITY_SENSOR/Z{m}S{n}_PDU_HUMI_{k}',
                    'TEMP_SENSOR/Z{m}S{n}_PDU_TEMP_{k}',
                    'SF/Z{m} PAHU {n}/SUP_TEMP']

data_hall_outputs = ['SF/Z{m} PAHU {n}/SUP_TEMP']

#
# param_dict = {
#     # "pue_full" : [pue_full_input, pue_full_output, 1],
#     # "pahu": [pahu_input_params, pahu_output_params],
#     # "data_hall": [data_hall_input_params, data_hall_output_params],
#     "chiller": [chiller_input_params, chiller_output_params]
# }

fill = lambda x, y, n: [i.format(**{y:n}) for i in x]
INP = "input"
OUT = "output"
NAME = "name"


# make CHILLERS =================
chillers = []
for m in [1, 2]:
    c = {INP: fill(chiller_input_params, "m", m),
     OUT: fill(chiller_output_params, "m", m),
     NAME: f"chiller_{m}"}
    chillers.append(c)


# make PAHU =================
pahus = []
for m in [1, 2]:
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        c = {INP: [p.format(m=m, n=n) for p in pahu_input_params],
         OUT: [p.format(m=m, n=n) for p in pahu_output_params],
         NAME: f"pahu_{m}_{n}"}

        pahus.append(c)

# make PUE =================
pues = []
_i = []
_o = []

data_hall_part = [{INP: _i, OUT: _o}]

for param in data_hall_inputs:
    _i.extend(expand_param(param))

for param in data_hall_outputs:
    _o.extend(expand_param(param))

# ================= DONE MAKING ======================

parts = {
    "chiller": chillers,
    "pahu": pahus,
    "data_hall": data_hall_part
    }

# system_map = {
#     chiller.input: pahu.output1,
#     chiller.output: pahu.input1,
#     data_hall.input: pahu.output2,
#     data_hall.output: pahu.input2,
# }

data_file = "/home/apurva/work/projects/dcool/data/flipkart/timesorted_serialized_dec.csv"
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from random import randint
import pandas as pd
from flask import make_response, redirect, url_for
import json
import gzip
from eplus_modelling.eplusmodel import EplusExperiment

app = Flask(__name__)
experiments = {}

class DFTimeSeriesGen:
    def __init__(self, df_csv_path):
        self.data = self.load_data(df_csv_path)
        self.index = 0
        self.columns = self.data.columns.tolist()

    def load_data(self, df_csv_path):
        df = pd.read_csv(df_csv_path)
        df.dropna(thresh=7, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0.0, inplace=True)

        # df = df.drop(columns=["timestamp"])
        return df

    def next(self):
        row = self.data.iloc[self.index]
        self.index = (self.index + 360) % len(self.data)
        return list(row.values)

    def row_as_dict(self, row):
        return dict(zip(self.columns, row))

    def header(self):
        # header = list(filter(lambda x: "ems" in x.lower() and ("pdu" not in x.lower()), self.columns))
        header = list(self.columns)
        return header

    def modifiable(self):
        # return list(filter(lambda x: "_sp" in x.lower()[-4:], self.columns))
        return list(self.columns)

class TimeSeriesGen:
    def __init__(self, num_cols):
        self.index = 0
        self.value = 0
        self.num_cols = num_cols
        self.cols = ["sensor_" + str(i) for i in range(1, self.num_cols + 1)]

    def modifiable(self):
        return self.cols
        
    def header(self):
        return self.cols

    def results(self):
        results = {}
        for i in range(120):
            for col in self.cols:
                results[col] = results.get(col, [])
                results[col].append(randint(10,20))
        return results

class ABExperimentTimeSeries:
    def __init__(self):
        self.eplus_exp = EplusExperiment("experiment_server")
        self.results_a = None
        self.index = 0
        self.columns = []

    def run_experiment(self, setpoints):
        self.eplus_exp.set_period(start=(1, 12), end=(10, 12))
        for setpoint in setpoints:
            self.eplus_exp.set_a(setpoint, str(setpoints[setpoint]))
        self.results_a = self.eplus_exp.run()
        self.columns = self.results_a.columns.to_list()
        params = [line.strip() for line in open('east_params.txt')]
        self.columns = params

    def modifiable(self):
        modifiables = [
          'oafractionsched',
          'cw:loop:temp:schedule',
          'chiller:alwaysonschedule',
          'tower:alwaysonschedule',
          'data:center:cpu:loading:schedule',
          'heating:setpoints',
          'cooling:setpoints',
        ]
        modifiables = set(modifiables)
        x = self.eplus_exp.get_modifiables()
        y = {}
        for e in x:
            e_ = e.replace(' ', ':')
            if e_ in modifiables:
                y[e_] = x[e]
        return y



    def header(self):
        return self.columns

    def results(self):
        res = {}
        for col in self.columns:
            row_a = list(map(float, list(self.results_a[col].values)))
            res[col] = row_a
        return res

# tsgen = TimeSeriesGen(9)
# tsgen = DFTimeSeriesGen("../outputs/flipkart/flat_df.csv")
# tsgen = DFTimeSeriesGen("../data/modelling/eplusout.csv")
# tsgen = DFTimeSeriesGen("../eplus_modelling/usual/eplusout.csv")

tsgen = ABExperimentTimeSeries()
print(tsgen.modifiable())
tsgen.run_experiment(setpoints=[])

@app.route('/run_experiment')
def run_experiment():
    setpoints = request.args
    setpoints = {x.replace(":",' '): setpoints[x] for x in setpoints}
    tsgen.run_experiment(setpoints)
    return {'status':'finished'}

@app.route('/')
def blank():
    return ''

@app.route('/ts')
def chart():
    content = tsgen.results()
    content = json.dumps(content).encode('utf8')
    response = make_response(content)
    response.headers['Content-length'] = len(content)
    response.headers['Content-Type'] = 'json'
    return response


@app.route('/tsinit')
def header():
    header = tsgen.header()
    modifiable = tsgen.modifiable()
    initinfo = {"header": header, "editable": modifiable}
    return initinfo

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()

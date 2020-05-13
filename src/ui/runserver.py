from flask import Flask
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
    
    def next(self):
        self.index += 1
        self.value = randint(18, 25)
        return [randint(0, 15) for col in self.cols]

class ABExperimentTimeSeries:
    def __init__(self):
        self.eplus_exp = EplusExperiment("experiment_server")
        self.results_a, self.results_b = None, None
        self.index = 0
        self.columns = []

    def run_experiment(self, setpoints_a, setpoints_b):
        self.eplus_exp.set_period(start=(1, 12), end=(5, 12))
        self.eplus_exp.set_ab("heating setpoints", str(15), str(20))
        self.eplus_exp.set_ab("cooling setpoints", str(18), str(35))
        self.results_a, self.results_b = self.eplus_exp.run()
        self.columns = self.results_b.columns.to_list()
        self.columns = list(filter(lambda x: "power" in x.lower(), self.columns))

    def modifiable(self):
        return self.eplus_exp.get_modifiables()

    def header(self):
        return self.columns

    def next(self):
        # print (self.index, self.results_b)
        row_a = self.results_a[self.columns].iloc[self.index]
        row_b = self.results_b[self.columns].iloc[self.index]
        self.index = (self.index + 1)
        tofloat = lambda l : list(map(float, list(l)))
        return [tofloat(row_a.values), tofloat(row_b.values)]

tsgen = TimeSeriesGen(9)
# tsgen = DFTimeSeriesGen("../outputs/flipkart/flat_df.csv")
# tsgen = DFTimeSeriesGen("../data/modelling/eplusout.csv")
# tsgen = DFTimeSeriesGen("../eplus_modelling/usual/eplusout.csv")

# tsgen = ABExperimentTimeSeries()

@app.route('/experiment')
def run_experiment(setpoints=None):
    tsgen.run_experiment(1, 2)
    return {'status':'finished'}

@app.route('/')
def blank():
    return ''

@app.route('/ts')
def chart():
    content = tsgen.next()
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

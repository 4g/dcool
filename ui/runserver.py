from flask import Flask
from gevent.pywsgi import WSGIServer
from random import randint
import pandas as pd
from flask import make_response
import json
import gzip

app = Flask(__name__)

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

class ABTimeSeries:
    def __init__(self, df_csv_path):
        self.experiment = self.create_experiment(a, b)
        self.index = 0
        self.columns = self.data.columns.tolist()

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

# tsgen = TimeSeriesGen(9)
# tsgen = DFTimeSeriesGen("../outputs/flipkart/flat_df.csv")
# tsgen = DFTimeSeriesGen("../data/modelling/eplusout.csv")
tsgen = DFTimeSeriesGen("../src/eplus_modelling/usual/eplusout.csv")


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

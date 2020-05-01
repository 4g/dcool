from flask import Flask
from gevent.pywsgi import WSGIServer
from random import randint
import pandas as pd
from flask import Response
import json

app = Flask(__name__)

class DFTimeSeriesGen:
    def __init__(self, df_csv_path):
        self.data = pd.read_csv(df_csv_path)
        self.index = 0
        self.columns = self.data.columns

    def next(self):
        row = self.data[self.index]
        self.index += 1
        return self.row_as_dict(row)

    def row_as_dict(self, row):
        return dict(zip(self.columns, row))

    def header(self):
        return dict(self.columns)

class TimeSeriesGen:
    def __init__(self, num_cols):
        self.index = 0
        self.value = 0
        self.num_cols = num_cols
        self.cols = ["ch" + str(i) for i in range(1, self.num_cols + 1)]
        
    def header(self):
        return self.cols
    
    def next(self):
        self.index += 1
        self.value = randint(18, 25)
        return {col: randint(0, 15) for col in self.cols}

tsgen = TimeSeriesGen(9)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ts')
def chart():
    return tsgen.next()


@app.route('/tsinit')
def header():
    header = tsgen.header()
    initinfo = {"header": header}
    return json.dumps(initinfo)

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()

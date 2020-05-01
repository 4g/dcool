from flask import Flask
from gevent.pywsgi import WSGIServer
from random import randint
import pandas as pd

app = Flask(__name__)

class DFTimeSeriesGen:
    def __init__(self, df_csv_path):
        self.data = pd.read_csv(df_csv_path)
        self.index = 0
        self.columns = self.data.columns

    def next(self):
        row = self.data[self.index]
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
        self.cols = ["sensor_" + str(i) for i in range(self.num_cols)]
        
    def header(self):
        return dict(self.cols)
    
    def next(self):
        self.index += 1
        self.value = randint(18, 25)
        return {col: randint(18, 25) for col in self.cols}

tsgen = TimeSeriesGen(10)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ts')
def chart():
    return tsgen.next()


@app.route('/tsinit')
def chart():
    return tsgen.header()

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()

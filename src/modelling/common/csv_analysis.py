import os
from pathlib import Path
import csv
from tqdm import tqdm
from datetime import datetime
import math
from file_utils import list_files

import matplotlib.pyplot as plt
import numpy as np

def header(csv_file):
    data = csv.reader(open(csv_file, errors='ignore'))
    h = next(data)
    print (h)
    return h

def data(csv_file, column, start, nrows):
    column = int(column)
    data = csv.reader(open(csv_file, errors='ignore'))
    h = next(data)
    values = [float(i[column]) for i in data]
    if nrows:
        values = values[start:start+nrows]
    return values

def plotm(files, column, start, nrows):
    plt.rcParams['figure.figsize'] = 10, 20
    names = set()
    for index, f in enumerate(files):
        values = data(f, column, start, nrows)
        plt.subplot(len(files), 1, index + 1)
        small_f = f.name
        if small_f in names:
            print (small_f)

        names.add(small_f)
        plt.title(small_f)
        plt.plot(values, 'b.')

    # plt.show()
    plt.savefig('test2png.png')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, required=True)

    parser.add_argument("--action", required=True)
    parser.add_argument("--column", required=False, default=0)
    parser.add_argument("--start", required=False, default=0)
    parser.add_argument("--nrows", required=False, default=-1)

    args = parser.parse_args()

    if args.action.lower() == "header":
        header(args.csv)

    if args.action.lower() == "plotm":
        files = []
        if os.path.isdir(args.csv):
            for dir_path, file_name in list_files(args.csv):
                file_path = dir_path / file_name
                files.append(file_path)

        elif os.path.isfile(args.csv):
            files = [args.csv]

        plotm(files,
              column=int(args.column),
              start=int(args.start),
              nrows=int(args.nrows))
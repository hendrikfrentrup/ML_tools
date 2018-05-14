import csv
from matplotlib import pylab as pl

#load csv
data_file='data.csv'
data=[]
with open(data_file, 'r') as csvfile:
    line = csv.reader(csvfile, delimiter=',')
    for l in line:
        data.append((l[0],l[1]))


#def func(x, b, w):

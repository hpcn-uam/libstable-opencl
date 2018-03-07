#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np

infile = sys.argv[1]

table = pd.read_table(infile, sep=" ", skiprows=300)

num_params = 4

ncols = table.shape[1]
print(ncols)
print(table.shape)
ncomps = (ncols - 2) / (num_params + 1)

max_lag = 5000
lags = []
headers = []

for comp in range(ncomps):
	for param in range(num_params):
		print comp, param
		data = table.iloc[:, 2 + comp * (num_params + 1) + param]
		lag = np.array([data.autocorr(lag=n) for n in range(max_lag)])
		lags.append(lag)
		print "done"

lags = np.array(lags).T

np.savetxt('mixture_autocorr.dat', lags)


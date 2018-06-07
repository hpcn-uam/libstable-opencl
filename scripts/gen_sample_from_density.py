import numpy as np
import sys

data = np.loadtxt(sys.argv[1])
nsamples = int(sys.argv[2])

mn = np.min(data[:, 0])
mx = np.max(data[:, 0])
mx_y = np.max(data[:, 1])

gensamples = 0
ntries = 0

while gensamples < nsamples:
	rnd_x = np.random.uniform(mn, mx)
	rnd_y = np.random.uniform(0, mx_y)
	val_y = np.interp(rnd_x, data[:, 0], data[:, 1], left=0, right=0)

	if val_y > rnd_y:
		print rnd_x
		gensamples += 1

	ntries += 1

print >> sys.stderr, "{0} tries to generate {1} samples".format(ntries, nsamples)

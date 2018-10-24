import scipy.io
import numpy as np

data = scipy.io.loadmat("target.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt((i+".csv"),data[i],delimiter=',')
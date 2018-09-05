import numpy as np
import pickle
import copy

pklfile = "flower.pkl"
f = open('../pkl/' + pklfile, 'rb')
data = pickle.load(f)
f.close()

aug = "flipLR_"
data2 = copy.deepcopy(data)
for key, value in  data.iteritems():
    keyname = aug + key
    new = np.empty([0, 5])
    for v in value:
        flip = np.array([1 - v[2], v[1], 1 - v[0], v[3], v[4]])
        flip = flip.reshape(1, 5)
        new = np.r_[new, flip]
    data2.update({keyname: new})

pickle.dump(data2,open('../pkl/flip_' + pklfile,'wb'))


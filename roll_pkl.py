import numpy as np
import pickle
import copy,math


def rollpoint(data,deg):
    rolled = []

    ad = data[0] - 0.5
    bd = data[1] - 0.5
    cx = ad * math.cos(deg) + bd * math.sin(deg) + 0.5
    cy = -ad * math.sin(deg) + bd * math.cos(deg) + 0.5
    ad = data[2] - 0.5
    bd = data[3] - 0.5
    cx1 = ad * math.cos(deg) + bd * math.sin(deg) + 0.5
    cy1 = -ad * math.sin(deg) + bd * math.cos(deg) + 0.5

    return [cx,cy,cx1,cy1]

pklfile = "flip_flower.pkl"
f = open('../pkl/' + pklfile, 'rb')
data = pickle.load(f)
f.close()

aug = "roll90_"
data2 = copy.deepcopy(data)
for key, value in  data.iteritems():
    keyname = aug + key
    new = np.empty([0, 5])
    for v in value:
        roll = np.array(rollpoint(v,(90/180.0)*math.pi))
        roll = np.append(roll, v[4])
        roll = roll.reshape(1, 5)
        new = np.r_[new, roll]
    data2.update({keyname: new})

pickle.dump(data2,open('../pkl/roll90_' + pklfile,'wb'))


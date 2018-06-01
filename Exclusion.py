import pickle
import numpy as np

filename = "test"
f = open("pkl/" + filename + ".pkl", 'rb')
data = pickle.load(f)
f.close()

rec = 0.001004

def calcRectangle(loc):
    x =  loc[2] - loc[0]
    y = loc[3] - loc[1]
    return x * y


for key , value in data.iteritems():
    tmp = np.array([])
    for i, v in enumerate(value):
        area = calcRectangle(v)
        if area < rec:
            continue
            np.delete(data[key], i, 0)
        tmp = np.append(tmp,data[key][i])

    tmp = tmp.reshape(len(tmp)/5,5)
    data[key] = tmp

pickle.dump(data,open("pkl/"  + filename + str(rec) +'.pkl','wb'))

f = open("pkl/" + filename + str(rec) +'.pkl', 'rb')
data = pickle.load(f)
f.close()

count = 0

for value in data.values():
  for v in value:
	count += 1

print count
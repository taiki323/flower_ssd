import pickle

pk = "nougi_flower3.pkl" #1228
pk = "crop_flower.pkl"
pkl = "pkl/" + pk
f = open(pkl, 'rb')
data = pickle.load(f)
f.close()

count = 0

for value in data.values():
  for v in value:
	count += 1

print count
print len(data.keys())

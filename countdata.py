import pickle
f = open('flower0.0015.pkl', 'rb')
data = pickle.load(f)
f.close()

count = 0

for value in data.values():
  for v in value:
	count += 1

print count

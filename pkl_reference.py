import pickle
f = open('flower0.002.pkl', 'rb')
data = pickle.load(f)
f.close()
print(data.keys())
print(data['007571.jpg'])

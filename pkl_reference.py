import pickle
f = open('flower.pkl', 'rb')
data = pickle.load(f)
print(data.keys())
print(data['007571.jpg'])

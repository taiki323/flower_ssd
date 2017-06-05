import pickle
f = open('flower89.pkl', 'rb')
data = pickle.load(f)
print(data.keys())
print(data['007571.jpg'])

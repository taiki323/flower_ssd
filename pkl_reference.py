import pickle
f = open('pkl/crop_flower.pkl', 'rb')
data = pickle.load(f)
f.close()
print(data.keys())
print(data['007571.jpg'])

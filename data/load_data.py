import pickle
file = open('bible.txt', 'rb')
data = pickle.load(file)
print(data[0][0])

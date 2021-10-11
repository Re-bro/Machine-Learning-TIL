from K_NN_data import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

## draw data
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14

## train and test
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
print(kn.score(fish_data, fish_target)) # 1


################ Divide train set & test set ##################


## make data
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

## shuffle data
np.random.seed(42)
index = np.arange(49) # [0, 1, 2, ... , 48]
np.random.shuffle(index)

## devide
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

## draw data
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_input, train_target)
print(kn.score(test_input, test_target)) # 1.0

print(kn.predict(test_input)) # [0 0 1 0 1 1 1 0 1 1 0 1 1 0]
print(test_target) # [0 0 1 0 1 1 1 0 1 1 0 1 1 0]


#################### Wrong answer by different scale ###################


print(kn.predict([[25.0, 150.0]])) # [0]

plt.scatter(input_arr[:,0], input_arr[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

distances, indexes = kn.kneighbors([[25, 150]])
print(distances) # [[ 92.00086956 130.73859415 137.17988191 138.32150953 138.39320793]]


##################### z-score standardization ######################


mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std) # [ 28.29428571 483.35714286] [  9.54606704 323.47456715]
train_scaled = (train_input - mean) / std

new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target) # 1.0
print(kn.predict([new])) # [1]

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


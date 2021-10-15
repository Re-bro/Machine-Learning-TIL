from data import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

print(knr.score(test_input, test_target)) # 0.992809406101064
print(knr.score(train_input, train_target)) # 0.9698823289099254


##################### solve underfitting problem ####################

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target)) # 0.9804899950518966
print(knr.score(test_input, test_target)) # 0.9746459963987609


##################### Limitation of K-NN Regression ###################

print(knr.predict([[100]])) # [1033.33333333]

distances, indexes = knr.kneighbors([[100]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes])
plt.scatter(100, 1033, marker='^')
plt.show()
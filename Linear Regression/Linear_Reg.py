from data import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[100]])) # [3192.69585141]
print(lr.coef_, lr.intercept_) # [39.01714496] -709.018644953547

plt.scatter(train_input, train_target)
plt.plot([15, 100], [15*lr.coef_ + lr.intercept_, 100*lr.coef_ + lr.intercept_])
plt.scatter(100, 3192, marker='^')
plt.show()

print(lr.score(train_input, train_target)) # 0.9398463339976041
print(lr.score(test_input, test_target)) # 0.8247503123313562


###################### multiple regression #######################


train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.coef_, lr.intercept_) # [  1.01433211 -21.55792498] 116.05021078278259
print(lr.predict([[100**2, 100]])) # [8103.57880667]

point = np.arange(15, 100)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point +  116.05)
plt.scatter([100], [8103], marker='^')
plt.show()

print(lr.score(train_poly, train_target)) # 0.9706807451768623
print(lr.score(test_poly, test_target)) # 0.9775935108325121

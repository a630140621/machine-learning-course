import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path
from sklearn.linear_model import LinearRegression


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex1data2.csv"), names=("size of house", "number of bedrooms", "price"))
# 训练集
X = data.loc[:,['size of house', 'number of bedrooms']]
# label
y = data["price"]
# ones = np.ones((X.shape[0], 1))
# X = np.concatenate((ones, np.array(X)), axis=1)

# model
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

# predict
# 2609,4,499998
print(f'For size of house = 2609, number of bedrooms = 4 we predict price equal {reg.predict([[2609, 4]])}')
# 3031,4,599000
print(f'For size of house = 3031, number of bedrooms = 4 we predict price equal {reg.predict([[3031, 4]])}')
# 1767,3,252900
print(f'For size of house = 1767, number of bedrooms = 3 we predict price equal {reg.predict([[1767, 3]])}')
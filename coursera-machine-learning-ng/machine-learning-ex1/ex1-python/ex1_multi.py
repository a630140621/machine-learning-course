import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from os import path
from sklearn.linear_model import LinearRegression


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex1data2.csv"), names=(
    "size of house", "number of bedrooms", "price"))
# 训练集
X = data.loc[:, ['size of house', 'number of bedrooms']]
# label
y = data["price"]

# 可视化, x: size of house, y: number of bedrooms, z: price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.subplot(111, projection='3d') # 相同
ax.scatter(X["size of house"], X["number of bedrooms"], y, c="g")
ax.set_zlabel('price')
ax.set_ylabel('number of bedrooms')
ax.set_xlabel('size of house')
plt.show()

# model
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

# predict
# 2609,4,499998
print(
    f'For size of house = 2609, number of bedrooms = 4 we predict price equal {reg.predict([[2609, 4]])}')
# 3031,4,599000
print(
    f'For size of house = 3031, number of bedrooms = 4 we predict price equal {reg.predict([[3031, 4]])}')
# 1767,3,252900
print(
    f'For size of house = 1767, number of bedrooms = 3 we predict price equal {reg.predict([[1767, 3]])}')

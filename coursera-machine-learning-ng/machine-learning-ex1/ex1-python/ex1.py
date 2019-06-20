import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path
from sklearn.linear_model import LinearRegression


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex1data1.csv"), names=("profit", "population"))
# 训练集
X = data["profit"]
# label
y = data["population"]

# 绘图(可视化)
# plt.scatter(X, y, marker="X", c="red", linewidths=0.1)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
# plt.show()

# 拟合数据
## theta.shape=(2, 1)
# theta = np.zeros((2, 1))
## X.shape=(97, 2)
ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, np.array(X).reshape(-1, 1)), axis=1)

# train
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
# 可视化
# plt.plot(y, reg.predict(X))
# plt.show()

# predict
# Predict values for population sizes of 35,000 and 70,000
print(f'For population = 35,000, we predict a profit of {reg.predict([[1, 3.5]]) * 10000}')
print(f'For population = 70,000, we predict a profit of {reg.predict([[1, 7]]) * 10000}')
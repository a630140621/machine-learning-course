import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import path
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def plotScatter(X, y):
    plt.figure()
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    plt.scatter(X_pos["p1"], X_pos["p2"], marker="x", c="r")
    plt.scatter(X_neg["p1"], X_neg["p2"], marker="o", c="b")
    plt.show()


# 随机生成数据
def randomGenData():
    number = 400
    X = np.random.rand(number, 2) * 2 - 1  # 随机生成 [-1, 1] 随机数
    D_X = pd.DataFrame(X, columns=["p1", "p2"])
    # 添加更多的featured, 在添加这些featured之前, 直接使用逻辑回归模型效果很差
    D_X["p1*p1"] = D_X["p1"] ** 2
    D_X["p1*p2"] = D_X["p1"] * D_X["p2"]
    D_X["p2*p2"] = D_X["p2"] ** 2
    return D_X


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex2data2.csv"),
                   names=("p1", "p2", "y"))

data = shuffle(data)

X_train = data.iloc[:, 0: 2]
y_train = data.iloc[:, 2]

# 添加更多的featured, 在添加这些featured之前, 直接使用逻辑回归模型效果很差
X_train["p1*p1"] = X_train["p1"] ** 2
X_train["p1*p2"] = X_train["p1"] * X_train["p2"]
X_train["p2*p2"] = X_train["p2"] ** 2

print(X_train.shape, y_train.shape)

# X_test = data.iloc[100:, 0: 2]
# y_test = data.iloc[100:, 2]
# print(X_test.shape, y_test.shape)


plotScatter(X_train, y_train)


# model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# linear svm model
# from sklearn.svm import LinearSVC
# model = LinearSVC()
# model.fit(X_train, y_train)

# NuSVC
# from sklearn.svm import NuSVC
# model = NuSVC(gamma='scale')
# model.fit(X_train, y_train)

# SVC
# from sklearn.svm import SVC
# model = SVC(gamma='auto')
# model.fit(X_train, y_train)


# print(model.classes_)
# print(model.coef_)
# print(model.intercept_)
# print(model.n_iter_)

# plot
# decision_function
# print(model.decision_function(X_test))


# plot predict
X_test = randomGenData()
# y_pre = model.decision_function(X_test)

# print(y_pre)
# y_pre[y_pre >= 0] = 1
# y_pre[y_pre < 0] = 0

# # print(y_test)
# print(y_pre)

y_pre = model.predict(X_test)

plotScatter(X_test, y_pre)

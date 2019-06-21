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


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex2data2.csv"),
                   names=("p1", "p2", "y"))

data = shuffle(data)

X_train = data.iloc[:100, 0: 2]
y_train = data.iloc[:100, 2]
print(X_train.shape, y_train.shape)

X_test = data.iloc[100:, 0: 2]
y_test = data.iloc[100:, 2]
print(X_test.shape, y_test.shape)


plotScatter(X_train, y_train)


# model
model = LogisticRegression(solver="liblinear", verbose=0)
model.fit(X_train, y_train)


# print(model.classes_)
# print(model.coef_)
# print(model.intercept_)
# print(model.n_iter_)

# plot
# 随机生成数据
def randomGenData():
    number = 400
    X = np.random.rand(number, 2) * 2 - 1  # 随机生成 [-1, 1] 随机数
    D_X = pd.DataFrame(X, columns=["p1", "p2"])
    return D_X


# decision_function
# print(model.decision_function(X_test))


# plot predict
X_test = randomGenData()
# y_decision = model.decision_function(X_test)

# print(y_decision)
# y_decision[y_decision >= 0] = 1
# y_decision[y_decision < 0] = 0

# print(y_test)
# print(y_decision)

y_pre = model.predict(X_test)

plotScatter(X_test, y_pre)

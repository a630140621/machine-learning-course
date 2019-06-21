import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os import path
from sklearn.linear_model import LogisticRegression


# 读取数据集
data = pd.read_csv(path.join(os.getcwd(), "ex2data1.csv"),
                   names=("p1", "p2", "y"))

X = data.loc[:, ["p1", "p2"]]
y = data["y"]

# plot
plt.figure(num=1)

X_pos = X[y == 1]
X_neg = X[y == 0]

# 正样本为 + 负样本为 ·
plt.scatter(X_pos["p1"], X_pos["p2"], marker="x", c="r")
plt.scatter(X_neg["p1"], X_neg["p2"], marker="o", c="b")
# plt.show()

# model
model = LogisticRegression(solver="liblinear", verbose=0)
model.fit(X, y)

# predict
print(
    f'class = {model.classes_} corresponding probability = {model.predict_proba([[45, 85]])}')
print(
    f'For a student with scores 45 and 85, we predict an admission probability of {model.predict([[45, 85]])}')
print('Expected value: 0.775 +/- 0.002')


# 随机生成数据
def randomGenData():
    number = 400
    X = np.random.rand(number, 2) * 100
    X_out = X[X[:, 0] > 30]
    X_out = X_out[X_out[:, 1] > 30]
    return X_out


# plot predict
X_test = randomGenData()
y_pre = model.predict(X_test)

plt.figure(num=2)

X_test_pos = X_test[y_pre == 1]
X_test_neg = X_test[y_pre == 0]

# 正样本为 + 负样本为 ·
plt.scatter(X_test_pos[:, 0], X_test_pos[:, 1], marker="x", c="r")
plt.scatter(X_test_neg[:, 0], X_test_neg[:, 1], marker="o", c="b")
plt.show()

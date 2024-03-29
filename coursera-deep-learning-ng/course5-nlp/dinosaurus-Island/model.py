# 该模型模仿自 keras/examples/lstm_text_generation.py 和吴恩达老师课程中讲的并不相同
# 吴恩达老师的模型是训练一个RNN模型，然后每一次输入一个单词，预测出下一个最可能的单词作为输出
# 此模型则是利用 corpus 构建一个监督学习模型，模型的构成为，选择一个当前字母作为x，下一个字母作为y，从而构建模型

# 当前采用了每一个单词之后使用 \n 补齐，对模型效果可能会有点影响
from os import path
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
import numpy as np
# from keras.preprocessing.sequence import pad_sequences


# 提供的不能用，因为 split 不支持 ''
# one = one_hot("t e x t", 27, lower=True, split=' ')
# print(one)

# 读取训练集
def get_train_data():
    dataset_path = path.join(path.dirname(__file__), "dataset.csv")
    with open(dataset_path, "r") as f:
        dataset = f.read()

    chars = list(set(dataset.lower()))
    dataset = dataset.split('\n')
    # 找到长度最长的名字
    maxlen = len(max(dataset))
    # 使用 \n 填充长度短于 maxlen 的名字
    dataset = [item.ljust(maxlen, '\n') for item in dataset]
    return dataset, chars, maxlen


dataset, chars, maxlen = get_train_data()
vocab_size = len(chars)
print(f'There are {len(dataset)} total names and {len(chars)} unique characters in your data.')


# embeding
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
# print(char_to_ix)
# print(ix_to_char)


# def word_to_one_hot(word):
#     # 将单词转换成 one-hot
#     one_hot = []
#     for w in word:
#         zeros = np.zeros((vocab_size, 1))
#         zeros[char_to_ix[w]] = 1
#         one_hot.append(zeros)

#     return one_hot


# def one_hot_to_word(one_hot):
#     # 将 one-hot 转换成单词
#     word = ''
#     for one in one_hot:
#         # 找到 word index
#         index = 0
#         for i in one:
#             if i[0] == 1:
#                 word += ix_to_char[index]
#             index += 1

#     return word


# print(word_to_one_hot("text"))
# print(one_hot_to_word(word_to_one_hot("text")))

# build model


def build():
    model = Sequential()
    # model.add(Embedding(len(chars) + 1, 64, input_length=10))
    # 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
    # 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
    # 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

    model.add(LSTM(128))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    # model.fit(x_train, y_train, batch_size=16, epochs=10)
    # score = model.evaluate(x_test, y_test, batch_size=16)

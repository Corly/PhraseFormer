from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Flatten, Input,
                                     Lambda, Reshape, TimeDistributed, Masking)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import tensorflow as tf

model = Sequential()
model.add(Masking(mask_value=[-10] * 896, input_shape=(None, 768 + 128)))
model.add(Bidirectional(LSTM(units=32, return_sequences=True))) # self.max_sequence_len,
model.add(TimeDistributed(Dense(units=8, activation="relu")))
model.add(TimeDistributed(Dense(units=3, activation="softmax")))
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

input = [[[0] * 896] * 15, [[0] * 896] * 20]
input = tf.keras.preprocessing.sequence.pad_sequences(input, padding="post", value=[-10] * 896)
input = np.array(input)
output = [[0] * 15 + [-10] * 5, [0] * 20]
output = np.array(output)
# with open("ex_input.pkl", "rb") as f:
#     input = pickle.load(f)
# with open("ex_output.pkl", "rb") as f:
#     output = pickle.load(f)


# input = input.tolist()
# # input[0] = list(input[0])
# input = np.asarray(input).astype(np.float32)
# modified = input.astype(np.float32)
# modified = input.tolist()
# modified = np.asarray(modified).astype(np.float32)
# print(input.shape, modified.shape)
# print(input.dtype, modified.dtype)
# print(np.array_equal(input, modified))
# print(np.array_equal(input[0], modified[0]))
# print(type(input), type(input[0]), type(input[0][0]), type(input[0][0][0]))
# type_set = set()
# for i in input[0][0]:
#     type_set.add(type(i))
# print(type_set)
# for i in input:
#     print(type(i))
#     for x in i:
#         print(x)
#         print(type(x))
#         exit()
# exit()

model.fit(input, output, epochs=10, verbose=1)
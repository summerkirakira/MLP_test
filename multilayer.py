import numpy as np
from .single_layer import load_idx1_ubyte, load_idx3_ubyte, sigmoid


layer_1 = np.array([])
b_1 = np.array([])
layer_2 = np.array([])
b_2 = np.array([])
layer_3 = np.array([])
b_3 = np.array([])


def computation(input):
    global layer_1, layer_2, layer_3, b_1, b_2, b_3
    a_1 = input @ layer_1 + b_1.T
    a_1 = sigmoid(a_1)
    a_2 = a_1 @ layer_2 + b_2.T
    a_2 = sigmoid(a_2)
    a_3 = a_2 @ layer_3 + b_3.T
    a_3 = sigmoid(a_3)


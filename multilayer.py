import numpy as np
from single_layer import load_idx1_ubyte, load_idx3_ubyte, sigmoid
from pathlib import Path
from matplotlib import pyplot as plt
import time


layer_1 = np.array([])
b_1 = np.array([])
layer_2 = np.array([])
b_2 = np.array([])
layer_3 = np.array([])
b_3 = np.array([])


def d_sigmoid(x):
    return x * (1 - x)


def compute_gradient(input, expect_output):
    global layer_1, layer_2, layer_3, b_1, b_2, b_3
    a_1 = input @ layer_1 + b_1
    a_1 = sigmoid(a_1)
    a_2 = a_1 @ layer_2 + b_2
    a_2 = sigmoid(a_2)
    a_3 = a_2 @ layer_3 + b_3
    a_3 = sigmoid(a_3)

    delta_3 = a_3 - expect_output
    delta_2 = delta_3 @ layer_3.T * d_sigmoid(a_2)
    delta_1 = delta_2 @ layer_2.T * d_sigmoid(a_1)

    total_b_3_gradient = delta_3 * d_sigmoid(a_3)
    total_b_2_gradient = delta_2 * d_sigmoid(a_2)
    total_b_1_gradient = delta_1 * d_sigmoid(a_1)

    total_w_3_gradient = a_2.T @ total_b_3_gradient
    total_w_2_gradient = a_1.T @ total_b_2_gradient
    total_w_1_gradient = input.T @ total_b_1_gradient

    a = layer_1
    b = layer_2
    c = layer_3

    return total_w_1_gradient, total_w_2_gradient, total_w_3_gradient, \
        total_b_1_gradient, total_b_2_gradient, total_b_3_gradient, delta_3


alpha = 1.5e-4


def predict(input) -> int:
    global layer_1, layer_2, layer_3, b_1, b_2, b_3
    a_1 = input @ layer_1 + b_1
    a_1 = sigmoid(a_1)
    a_2 = a_1 @ layer_2 + b_2
    a_2 = sigmoid(a_2)
    a_3 = a_2 @ layer_3 + b_3
    a_3 = sigmoid(a_3)
    return np.argmax(a_3)


def main():
    global layer_1, layer_2, layer_3, b_1, b_2, b_3
    train_data_num = 60000
    raw_train_images = load_idx3_ubyte(Path('data/MNIST/raw/train-images-idx3-ubyte'))
    raw_train_labels = load_idx1_ubyte(Path('data/MNIST/raw/train-labels-idx1-ubyte'))

    test_images = load_idx3_ubyte(Path('data/MNIST/raw/t10k-images-idx3-ubyte'))
    test_labels = load_idx1_ubyte(Path('data/MNIST/raw/t10k-labels-idx1-ubyte'))

    train_images = raw_train_images[:train_data_num]
    train_labels = raw_train_labels[:train_data_num]

    train_images = train_images.astype(float) / 255.0
    test_images = test_images.astype(float) / 255.0

    expect_output = np.zeros((train_data_num, 10))
    for i in range(train_data_num):
        expect_output[i][train_labels[i]] = 1

    np.random.seed(11451)

    layer_1 = np.random.rand(784, 100) - 0.5
    b_1 = np.random.rand(1, 100) - 0.5
    layer_2 = np.random.rand(100, 20) - 0.5
    b_2 = np.random.rand(1, 20) - 0.5
    layer_3 = np.random.rand(20, 10) - 0.5
    b_3 = np.random.rand(1, 10) - 0.5

    iter_num = 2000

    for i in range(iter_num):
        start_time = time.time()
        total_w_1_gradient, total_w_2_gradient, total_w_3_gradient, \
            total_b_1_gradient, total_b_2_gradient, total_b_3_gradient, \
            error = compute_gradient(train_images, expect_output)
        layer_1 -= alpha * total_w_1_gradient
        layer_2 -= alpha * total_w_2_gradient
        layer_3 -= alpha * total_w_3_gradient
        b_1 -= alpha * np.sum(total_b_1_gradient, axis=0)
        b_2 -= alpha * np.sum(total_b_2_gradient, axis=0)
        b_3 -= alpha * np.sum(total_b_3_gradient, axis=0)
        error = np.sum(np.square(error))
        end_time = time.time()
        predict_time_remain = (end_time - start_time) * (iter_num - i)
        print(f'iter {i} finished, error: {error}, remain: {predict_time_remain} seconds')


    error_num = 0
    error_list = []
    for index, image in enumerate(test_images):
        predict_number = predict(image)
        if test_labels[index] != predict_number:
            error_num += 1
            error_list.append({
                'index': index,
                'error': predict_number,
                'expect': test_labels[index]
            })
            print(f'found: {predict_number} expect: {test_labels[index]}')

    print(f'error rate: {error_num / len(test_images)}')

    fig, axes = plt.subplots(5, 5)

    image_to_be_shown = np.random.choice(error_list, 25)
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_images[image_to_be_shown[i]['index']].reshape((28, 28)), cmap='gray')
        ax.axis('off')
        ax.set_title(f"{image_to_be_shown[i]['error']} -> {image_to_be_shown[i]['expect']}")

    plt.suptitle(f'Multilayer perceptron\nTrain examples: {train_data_num}, Iter number: {iter_num}, Error rate: {error_num / len(test_images)}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

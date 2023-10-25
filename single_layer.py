import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


def load_idx3_ubyte(filename: Path):
    with filename.open('rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.empty((num, rows * cols), dtype=np.uint8)
        for i in range(num):
            for row in range(rows):
                for col in range(cols):
                    images[i][row * rows + col] = int.from_bytes(f.read(1), 'big')
    return images


def load_idx1_ubyte(filename: Path):
    with filename.open('rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        labels = np.empty((num,), dtype=np.uint8)
        for i in range(num):
            labels[i] = int.from_bytes(f.read(1), 'big')
    return labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data():
    raw_train_images = load_idx3_ubyte(Path('data/handwritten_digits/train-images.idx3-ubyte'))
    raw_train_labels = load_idx1_ubyte(Path('data/handwritten_digits/train-labels.idx1-ubyte'))

    test_images = load_idx3_ubyte(Path('data/handwritten_digits/t10k-images.idx3-ubyte'))
    test_labels = load_idx1_ubyte(Path('data/handwritten_digits/t10k-labels.idx1-ubyte'))
    return raw_train_images, raw_train_labels, test_images, test_labels


if __name__ == '__main__':
    tran_data_num = 1000
    raw_train_images = load_idx3_ubyte(Path('data/handwritten_digits/train-images.idx3-ubyte'))
    raw_train_labels = load_idx1_ubyte(Path('data/handwritten_digits/train-labels.idx1-ubyte'))

    test_images = load_idx3_ubyte(Path('data/handwritten_digits/t10k-images.idx3-ubyte'))
    test_labels = load_idx1_ubyte(Path('data/handwritten_digits/t10k-labels.idx1-ubyte'))

    train_images = raw_train_images[:tran_data_num]
    train_labels = raw_train_labels[:tran_data_num]

    expect_output = np.zeros((tran_data_num, 10))
    for i in range(tran_data_num):
        expect_output[i][train_labels[i]] = 1

    train_images = train_images.astype(float) / 255.0

    iter_num = 1
    alpha = 5e-5

    np.random.seed(11451)

    weight_array = np.random.rand(784, 10) * 0.01 - 0.005
    b = np.random.rand(10, 1) * 0.01 - 0.005

    while iter_num < 5000:
        a_x = train_images @ weight_array + b.T
        y_x = sigmoid(a_x)
        delta = y_x - expect_output
        # a = 1 - y_x
        # a = alpha * (delta.T @ y_x @ (1 - y_x).T @ train_images).T
        total_b_gradient = delta * (1 - y_x) * y_x
        total_w_gradient = train_images.T @ total_b_gradient
        # total_b_gradient = np.zeros((1, 10))

        # for i in range(tran_data_num):
        #     x = train_images[i].reshape((784, 1))
        #     a = (delta[i] * (1 - y_x[i]) * y_x[i]).reshape((1, 10))
        #     total_b_gradient += a
        #     total_gradient += x @ a

        weight_array -= alpha * total_w_gradient
        b -= alpha * total_b_gradient.sum(axis=0).reshape((10, 1))
        print(f'iter_num: {iter_num} error: {np.sum(delta ** 2)}')
        iter_num += 1

    def predict(x):
        output = x @ weight_array + b.T
        return np.argmax(output, axis=1)

    error_num = 0
    error_list = []
    for index, image in enumerate(test_images):
        predict_number = predict(image)
        if test_labels[index] != predict_number[0]:
            error_num += 1
            error_list.append({
                'index': index,
                'error': predict_number[0],
                'expect': test_labels[index]
            })
            print(f'found: {predict_number[0]} expect: {test_labels[index]}')

    print(f'error rate: {error_num / len(test_images)}')

    fig, axes = plt.subplots(5, 5)

    image_to_be_shown = np.random.choice(error_list, 25)
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_images[image_to_be_shown[i]['index']].reshape((28, 28)), cmap='gray')
        ax.axis('off')
        ax.set_title(f"{image_to_be_shown[i]['error']} -> {image_to_be_shown[i]['expect']}")

    plt.suptitle(
        f'Single layer perceptron\nTrain examples: {tran_data_num}, Iter number: {iter_num}, Error rate: {error_num / len(test_images)}')
    plt.tight_layout()
    plt.show()
    # print(train_images.shape)

    # plt.imshow(train_images[600].reshape((28, 28)), cmap='gray')
    # plt.axis('off')
    # plt.title(f"Label: {train_labels[600]}")
    # plt.show()

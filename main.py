import numpy as np
from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt

data_path = Path('data/single')


def convert_text_to_array(path: Path) -> np.ndarray:
    data_set = []
    with path.open('r') as f:
        for line in f.readlines():
            for char in line:
                if char == '\n':
                    continue
                data_set.append(1 if char == '1' else 0)
    return np.array(data_set)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    matrix_1 = []
    matrix_0 = []
    for file in (data_path / '1').glob('*.txt'):
        matrix_1.append(convert_text_to_array(file))
    for file in (data_path / '0').glob('*.txt'):
        matrix_0.append(convert_text_to_array(file))
    return np.array(matrix_0), np.array(matrix_1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    data_set0, data_set1 = load_data()

    train_data_set = np.vstack((data_set0[:4, :], data_set1[:4, :]))
    test_data_set = np.vstack((data_set0[4:, :], data_set1[4:, :]))
    test_labels = [-1] * 2 + [1] * 2

    labels = [-1] * 4 + [1] * 4

    correct_response = np.array(labels).reshape(-1, 1)

    b = np.random.rand() * 0.01

    weight_array = (np.random.rand(25, 1) - 0.5) * 0.01

    alpha = 0.025

    count = 0

    while True:
        response = train_data_set @ weight_array + b
        result = correct_response - np.sign(response)
        if not np.any(result != 0):
            break
        weight_array += alpha * (train_data_set.T @ result)
        b += np.sum(alpha * result)
        count += 1
        print(f"{count} try")

    fig, axes = plt.subplots(3, 4, figsize=(8, 8))

    for i in range(4):
        axes[0, i].imshow(train_data_set[i].reshape((5, 5)), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label: {labels[i]}")
        axes[1, i].imshow(train_data_set[i + 4].reshape((5, 5)), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Label: {labels[i + 4]}")

    for i in range(len(test_data_set)):
        response = test_data_set[i]  @ weight_array + b
        result = np.sign(response)
        axes[2, i].imshow(test_data_set[i].reshape((5, 5)), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(f"Label: {test_labels[i]} / {result[0]}")

    plt.show()

    # print(weight_array)
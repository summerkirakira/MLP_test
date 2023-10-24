import numpy as np
from pathlib import Path
from typing import Tuple

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
    return np.array(matrix_0), np.array(matrix_0)


if __name__ == '__main__':
    data_set0, data_set1 = load_data()

    train_data_set = np.vstack((data_set0[:3, :], data_set1[:3, :])).T
    labels = [-1] * 3 + [1] * 3

    correct_response = np.array(labels).reshape(-1, 1)

    b = np.random.rand() * 0.01

    weight_array = (np.random.rand(25, 1) - 0.5) * 0.01

    alpha = 0.025

    count = 0
    while True:
        response = train_data_set.T @ weight_array + b
        result = np.sign(response) - correct_response
        if not np.any(result != 0):
            break
        weight_array += alpha * train_data_set @ (correct_response - response)
        b += np.sum(alpha * (correct_response - response))
        count += 1
        print(f"{count} try")


    print(weight_array)
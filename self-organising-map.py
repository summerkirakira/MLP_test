import numpy as np
from matplotlib import pyplot as plt


def get_data():
    labels = [
        'Dove',
        'Hen',
        'Duck',
        'Goose',
        'Owl',
        'Hawk',
        'Eagle',
        'Fox',
        'Dog',
        'Wolf',
        'Cat',
        'Tiger',
        'Lion',
        'Horse',
        'Zebra',
        'Cow'
    ]
    data = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ])
    return labels, data


def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def mu(i, j, u, v):
    return np.exp(-1 * ((i - u) ** 2 + (j - v) ** 2) / 8)

def main():
    labels, data = get_data()

    som = np.zeros((10, 10, 13))
    for i in range(10):
        for j in range(10):
            random_index = np.random.randint(0, len(data))
            som[i, j] = data[random_index]

    for t in range(10000):
        random_index = np.random.randint(0, len(data))
        x = data[random_index]

        min_distance = 1e10
        min_i_index = 0
        min_j_index = 0

        for i in range(10):
            for j in range(10):
                current_distance = distance(x, som[i, j])
                if current_distance < min_distance:
                    min_distance = current_distance
                    min_i_index = i
                    min_j_index = j

        for i in range(10):
            for j in range(10):
                som[i, j] += 100 / (200 + t) * mu(i, j, min_i_index, min_j_index) * (x - som[i, j])

        print(f'iter {t}')

    result_list = [[[] for _ in range(10)] for _ in range(10)]

    for k in range(len(data)):
        x = data[k]
        min_distance = 1e10
        min_i_index = 0
        min_j_index = 0

        for i in range(10):
            for j in range(10):
                current_distance = distance(x, som[i, j])
                if current_distance < min_distance:
                    min_distance = current_distance
                    min_i_index = i
                    min_j_index = j

        result_list[min_i_index][min_j_index].append(labels[k])
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].text(0.1, 0.5, '\n'.join(result_list[i][j]), fontsize=10)
            # axes[i, j].axis('off')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle('Self-Organising Map of Animals')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
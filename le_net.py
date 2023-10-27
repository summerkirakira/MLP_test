from single_layer import load_data
from torch import nn
import torch
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import random
import numpy as np
from pathlib import Path
import os

current_path = Path(os.path.dirname(__file__))


class MyLeNet5(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv4 = nn.AvgPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 120, 5)
        self.fc6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.conv2(x)
        x = self.sigmoid(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.output(x)
        return x


def get_data():
    # raw_train_images, raw_train_labels, raw_test_images, raw_test_labels = load_data()
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ])
    train_data = datasets.MNIST('data', train=True, transform=data_transform, download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    test_data = datasets.MNIST('data', train=False, transform=data_transform, download=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    return train_data_loader, test_data_loader, train_data, test_data


def train(model, train_data_loader, loss_func, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss, acc, n = 0, 0, 0
    for batch_idx, (data, target) in enumerate(train_data_loader):
        X, y = data.to(device), target.to(device)
        output = model(X)
        cur_loss = loss_func(output, y)
        _, pred = torch.max(output, 1)

        cur_acc = torch.sum(pred == y) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        acc += cur_acc.item()
        n += 1
    print(f"train loss: {loss / n}, train acc: {acc / n}")


def validate(model, test_data_loader, loss_func):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss, acc, n = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            X, y = data.to(device), target.to(device)

            output = model(X)
            cur_loss = loss_func(output, y)
            _, pred = torch.max(output, 1)

            cur_acc = torch.sum(pred == y) / output.shape[0]

            loss += cur_loss.item()
            acc += cur_acc.item()
            n += 1
        print(f"test loss: {loss / n}, test acc: {acc / n}")
    return acc / n


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyLeNet5().to(device)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_data_loader, test_data_loader, _, _ = get_data()

    epoch = 50

    min_acc = 0.0

    for i in range(epoch):
        print(f"epoch: {i + 1} \n {'-' * 20} \n")
        train(model, train_data_loader, loss_func, optimizer)
        a = validate(model, test_data_loader, loss_func)
        scheduler.step()
        if a > min_acc:
            min_acc = a
            with (current_path / 'model/lenet5.pth').open('wb') as f:
                torch.save(model.state_dict(), f)
            print('best model saved')
    print(f"best acc: {min_acc}")


def test():
    from matplotlib import pyplot as plt
    _, _, _, test_data = get_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fig, axes = plt.subplots(5, 5)

    err_list = []
    model = MyLeNet5().to(device)
    model.load_state_dict(torch.load(current_path / 'model/lenet5.pth'))
    label = [str(i) for i in range(10)]
    for i in range(len(test_data)):
        X, y = test_data[i]

        X = torch.autograd.Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)

        with torch.no_grad():
            pred = model(X)
            predicted, actual = label[torch.argmax(pred[0])], label[y]
            if predicted != actual:
                err_list.append((test_data[i][0].numpy(), predicted, actual))
                print(f"predicted: {predicted}, actual: {actual}")
            if i % 100 == 0:
                print(f"{i} images processed")

    image_to_be_shown = random.sample(err_list, 25)
    error_rate = len(err_list) / len(test_data)
    for i, ax in enumerate(axes.flat):
        ax.imshow(image_to_be_shown[i][0][0] * 255, cmap='gray')
        ax.set_xlabel(f"predicted: {image_to_be_shown[i][1]} \n actual: {image_to_be_shown[i][2]}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f'LeNet5\n Epoch: 25, Batch Size: 16, Learning Rate: 0.001, Error Rate: {error_rate}')
    plt.tight_layout()
    plt.savefig('temp.png')


if __name__ == '__main__':
    main()
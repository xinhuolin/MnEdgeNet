import torch
import random
import math
import numpy


class Model(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(Model, self).__init__()
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        self.lrate = lrate

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 13))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1, 2))

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 13))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 7))
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(1, 2))

        self.fc1 = torch.nn.Linear(41 * 128, 2048)

        self.fc2 = torch.nn.Linear(2048, 512)

        self.fc3 = torch.nn.Linear(512, self.out_size)

        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 1, 388))
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.dropout2(x)
        x = self.pool1(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = self.dropout2(x)
        x = self.pool3(x)

        x = x.view(-1, 41 * 128)

        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def step(self, x, y):
        result = self.forward(x)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrate)
        optimizer.zero_grad()
        loss = self.loss_fn(result, y)
        loss.backward()
        optimizer.step()
        return float(loss.item())


def fit(train_data, test_set, epoch, batch_size=32):
    # train_data is a nested list, test_set is already a torch tensor
    # the first 3 numbers of every row in train_data are the labels
    torch.autograd.set_detect_anomaly(True)
    model = Model(0.00008, torch.nn.MSELoss(), 388, 3).cuda()
    losses = []
    yhat = []
    index = 0

    train_set = train_data[:, 3:]
    train_set = torch.tensor(train_set, dtype=torch.float32)
    mean = train_set.mean(dim=0, keepdim=True)
    std = train_set.std(dim=0, keepdim=True)
    numpy.savetxt('EELSProject/mean.csv', mean.numpy(), delimiter=',')
    numpy.savetxt('EELSProject/std.csv', std.numpy(), delimiter=',')

    model.train()

    for i in range(epoch):
        print("Epoch: ", i)
        random.shuffle(train_data)
        train_label = train_data[:, 0:3]
        train_set = train_data[:, 3:]
        train_set = torch.tensor(train_set, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32).cuda()

        standardize = (train_set - mean) / std
        standardize = standardize.cuda()
        while 1:
            if index >= len(train_label):
                index = 0
                del standardize
                del train_label
                torch.cuda.empty_cache()
                break
            temp = model.step(standardize[index:index + batch_size], train_label[index:index + batch_size])
            losses.append(temp)
            index += batch_size
    torch.cuda.empty_cache()
    standardize = (test_set - mean) / std
    standardize = standardize.cuda()
    torch.save(model.state_dict(), 'EELSProject/EELS.pt')
    model.eval()
    index = 0
    while index < len(standardize):
        eval = model.forward(standardize[index:index+50])
        index += 50
        for item in eval:
            yhat.append(item.data.tolist())
    return losses, yhat, model


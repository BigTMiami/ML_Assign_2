import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset


class MNISTData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MNISTNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(784, 84)
        self.fc2 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        self.test_data = kwargs["test_data"]
        self.test_labels = kwargs["test_labels"]

        self.cv_data = None  # kwargs["cv_data"]
        self.cv_labels = None  # kwargs["cv_labels"]

        self.training_data_loader = kwargs["training_data_loader"]
        self.epoch_count = kwargs["epoch_count"]

        # For external training
        self.train_inputs, self.train_labels = next(iter(self.training_data_loader))
        self.state_length = 66790

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def check_accuracy(self, input, labels):
        output = self(input)
        _, predictions = t.max(output, 1)
        correct = (predictions == labels).sum().float()
        acc = 100 * correct / len(labels)
        return acc

    def check_test_accuracy(self):
        return self.check_accuracy(self.test_data, self.test_labels)

    def check_cv_accuracy(self):
        return 0
        return self.check_accuracy(self.cv_data, self.cv_labels)

    def get_state(self):
        fc1_weight = self.state_dict()["fc1.weight"]
        fc1_bias = self.state_dict()["fc1.bias"]
        fc2_weight = self.state_dict()["fc2.weight"]
        fc2_bias = self.state_dict()["fc2.bias"]

        state = t.cat((fc1_weight.flatten(), fc1_bias, fc2_weight.flatten(), fc2_bias))

        print(state.numpy()[66780:66790])

        return state.numpy()

    def load_state(self, state):
        load_dict = {}
        # fc1.weight: torch.Size([84, 784] 65856
        fc1_weight = state[0:65856]
        fc1_weight = t.tensor(np.reshape(np.array(fc1_weight), (84, 784)), dtype=t.float32)
        load_dict["fc1.weight"] = fc1_weight

        # fc1.bias torch.Size([84]) 65940
        fc1_bias = state[65856:65940]
        fc1_bias = t.tensor(np.array(fc1_bias), dtype=t.float32)
        load_dict["fc1.bias"] = fc1_bias

        # fc2.weight torch.Size([10, 84]) 66780
        fc2_weight = state[65940:66780]
        fc2_weight = t.tensor(np.reshape(np.array(fc2_weight), (10, 84)), dtype=t.float32)
        load_dict["fc2.weight"] = fc2_weight

        # fc2.bias torch.Size([10]) 66790
        fc2_bias = state[66780:66790]
        fc2_bias = t.tensor(np.array(fc2_bias), dtype=t.float32)
        load_dict["fc2.bias"] = fc2_bias

        self.load_state_dict(load_dict)

    def train(self):
        epoch_values = []
        iteration_values = []
        iteration_count = 0
        cv_acc = 0
        for epoch in range(self.epoch_count):  # loop over the dataset multiple times
            running_loss = 0.0
            for inputs, labels in self.training_data_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                test_acc = self.check_test_accuracy()
                iteration_values.append([iteration_count, loss, test_acc])
                print(f"Iter:{iteration_count:7} loss:{loss:10.4f} test acc:{test_acc:6.3f}")
                running_loss += loss.item()
                iteration_count += 1

            print("==============================================================")
            print(f"Epoch:{epoch:4} loss:{running_loss:10.4f} test acc:{test_acc:6.3f}")
            print("==============================================================")
            epoch_values.append([epoch, running_loss, cv_acc, test_acc])
        return epoch_values, iteration_values

    def get_state_loss(self):
        outputs = self(self.train_inputs)
        loss = self.criterion(outputs, self.train_labels)
        return loss

    def next_training_data(self):
        self.train_inputs, self.train_labels = next(iter(self.training_data_loader))

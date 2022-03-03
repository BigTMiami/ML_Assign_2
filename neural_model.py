import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from time import time

import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh


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

        self.train_acc_data = kwargs["train_acc_data"]
        self.train_acc_labels = kwargs["train_acc_labels"]

        self.cv_data = None  # kwargs["cv_data"]
        self.cv_labels = None  # kwargs["cv_labels"]

        self.training_data_loader = kwargs["training_data_loader"]
        self.epoch_count = kwargs["epoch_count"]

        # For external training
        # self.train_inputs, self.train_labels = next(iter(self.training_data_loader))
        self.train_inputs = None
        self.train_labels = None
        self.state_length = 66790

        # For randomized training
        self.fitness = mh.CustomFitness(self.get_loss_for_state, problem_type="continuous")
        self.neural_problem = mh.ContinuousOpt(
            self.state_length, self.fitness, maximize=False, min_val=-10, max_val=10, step=0.1
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def check_accuracy(self, input, labels):
        output = self(input)
        _, predictions = t.max(output, 1)
        correct = (predictions == labels).sum().float().item()
        acc = 100 * correct / len(labels)
        return acc

    def check_test_accuracy(self):
        return self.check_accuracy(self.test_data, self.test_labels)

    def check_cv_accuracy(self):
        return 0
        return self.check_accuracy(self.cv_data, self.cv_labels)

    def check_train_accuracy(self):
        return self.check_accuracy(self.train_acc_data, self.train_acc_labels)

    def get_state(self):
        fc1_weight = self.state_dict()["fc1.weight"]
        fc1_bias = self.state_dict()["fc1.bias"]
        fc2_weight = self.state_dict()["fc2.weight"]
        fc2_bias = self.state_dict()["fc2.bias"]

        state = t.cat((fc1_weight.flatten(), fc1_bias, fc2_weight.flatten(), fc2_bias))

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

    def train(self, capture_iteration_values=False):
        start_time = time()
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

                if capture_iteration_values:
                    test_acc = self.check_test_accuracy()
                    iteration_values.append([iteration_count, loss.item(), test_acc])
                    print(f"Iter:{iteration_count:7} loss:{loss:10.4f} test acc:{test_acc:6.3f}")
                running_loss += loss.item()
                iteration_count += 1

            test_acc = self.check_test_accuracy()
            train_acc = self.check_train_accuracy()
            print("==============================================================")
            print(f"Epoch:{epoch:4} loss:{running_loss:10.4f} test acc:{test_acc:6.3f} train acc:{train_acc:6.3f}")
            print("==============================================================")

            epoch_values.append([epoch, running_loss, train_acc, cv_acc, test_acc])

        training_time = time() - start_time
        print(f"Training time {training_time:.0f} seconds")
        return training_time, epoch_values, iteration_values

    def get_state_loss(self):
        outputs = self(self.train_inputs)
        loss = self.criterion(outputs, self.train_labels).item()
        return loss

    def next_training_data(self):
        self.train_inputs, self.train_labels = next(iter(self.training_data_loader))

    def get_loss_for_state(self, state):
        self.load_state(state)
        loss = self.get_state_loss()
        return loss

    def train_with_algorithm(self, algorithm_settings, capture_iteration_values=False):
        start_time = time()
        if algorithm_settings["algorithm"] not in ["rhc", "sa", "ga", "backprop"]:
            print(f"{algorithm_settings['algorithm']} not a support algorithm type")
            return None, None, None
        cv_acc = 0
        # old_acc = self.check_test_accuracy()
        # old_loss = self.get_state_loss()
        old_acc = 0
        old_loss = 0

        best_state = self.get_state()
        old_state = best_state
        epoch_values = []
        iteration_values = []
        iteration_count = 0
        # This primes the random state generator below for reproduceability
        np.random.seed(algorithm_settings["seed"])
        for epoch in range(algorithm_settings["epochs"]):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_start_time = time()
            i = 0

            all_change_counts = []
            if algorithm_settings["algorithm"] == "sa":
                if algorithm_settings["decay"] == "geom":
                    schedule = mh.GeomDecay(
                        init_temp=algorithm_settings["temperature"], min_temp=algorithm_settings["min_temp"]
                    )
                elif algorithm_settings["decay"] == "arith":
                    schedule = mh.ArithDecay(
                        init_temp=algorithm_settings["temperature"], min_temp=algorithm_settings["min_temp"]
                    )
                elif algorithm_settings["decay"] == "exp":
                    schedule = mh.ExpDecay(
                        init_temp=algorithm_settings["temperature"], min_temp=algorithm_settings["min_temp"]
                    )

            for inputs, labels in self.training_data_loader:
                self.train_inputs = inputs
                self.train_labels = labels
                if algorithm_settings["algorithm"] == "rhc":
                    best_state, best_fitness, _ = mh.random_hill_climb(
                        self.neural_problem,
                        max_attempts=algorithm_settings["max_attempts"],
                        max_iters=algorithm_settings["max_iters"],
                        restarts=algorithm_settings["restart"],
                        init_state=best_state,
                        curve=False,
                        random_state=np.random.randint(10000000),
                    )

                    self.load_state(best_state)
                    new_loss = self.get_state_loss()

                elif algorithm_settings["algorithm"] == "sa":

                    best_state, best_fitness, _ = mh.simulated_annealing(
                        self.neural_problem,
                        max_attempts=algorithm_settings["max_attempts"],
                        max_iters=algorithm_settings["max_iters"],
                        schedule=schedule,
                        init_state=best_state,
                        curve=False,
                        random_state=np.random.randint(10000000),
                    )

                    self.load_state(best_state)
                    new_loss = self.get_state_loss()

                elif algorithm_settings["algorithm"] == "ga":

                    best_state, best_fitness, _ = mh.genetic_alg(
                        self.neural_problem,
                        max_attempts=algorithm_settings["max_attempts"],
                        max_iters=algorithm_settings["max_iters"],
                        pop_size=algorithm_settings["population"],
                        mutation_prob=algorithm_settings["mutation"],
                        curve=False,
                        random_state=np.random.randint(10000000),
                    )

                    self.load_state(best_state)
                    new_loss = self.get_state_loss()

                elif algorithm_settings["algorithm"] == "backprop":
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    new_loss = loss.item()

                    # For calculatin parity
                    best_state = self.get_state()

                i += 1
                change_count = np.sum(old_state != best_state)
                all_change_counts.append(change_count)
                running_loss += new_loss
                old_state = best_state

                if capture_iteration_values:
                    new_acc = self.check_test_accuracy()

                    print(
                        f"{i}: Changes:{change_count:3} Acc:{new_acc:7.4f}  Acc Improvement:{new_acc - old_acc:10.6f} loss:{new_loss:10.4f} Loss improvement:{old_loss - best_fitness:10.6f} "
                    )
                    iteration_values.append([iteration_count, new_loss, new_acc])
                    old_acc = new_acc
                    old_loss = new_loss

            change_count_avg = sum(all_change_counts) / len(all_change_counts)
            weight_sum = np.sum(np.abs(best_state))
            epoch_time = time() - epoch_start_time
            test_acc = self.check_test_accuracy()
            train_acc = self.check_train_accuracy()
            print("==============================================================")
            print(
                f"Epoch:{epoch:4} loss:{running_loss:10.4f} test acc:{test_acc:6.3f} train acc:{train_acc:6.3f} time:{epoch_time:.0f}s change_count_avg:{change_count_avg:.0f} weight_sum:{weight_sum:.1f} "
            )
            print("==============================================================")

            epoch_values.append([epoch, running_loss, train_acc, cv_acc, test_acc, change_count_avg, weight_sum])

        training_time = time() - start_time
        print(f"Training time {training_time:.0f} seconds")
        return training_time, epoch_values, iteration_values

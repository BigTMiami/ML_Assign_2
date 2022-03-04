from mnist_data_prep import get_mnist_data_labels_neural
from neural_model import MNISTData, MNISTNet
from torch.utils.data import Dataset, DataLoader
import torch as t
import numpy as np

(
    train_images_flattened,
    train_one_hot_labels,
    train_labels,
    test_images_flattened,
    test_one_hot_labels,
    test_labels,
) = get_mnist_data_labels_neural()

mnist = MNISTData(train_images_flattened, train_one_hot_labels)
mnist_loader = DataLoader(mnist, batch_size=100, shuffle=True)


net = MNISTNet(
    training_data_loader=mnist_loader, test_data=test_images_flattened, test_labels=test_labels, epoch_count=10
)
net.check_test_accuracy()
results = net.train()

net.check_accuracy(test_images_flattened, test_labels)

total_values = 0
for key, value in net.state_dict().items():
    key_values = 1
    print(key)
    print("________________________________")
    print(value.shape)
    for i in value.shape:
        key_values = key_values * i
    print(key_values)
    total_values += key_values
    print(total_values)
print(f"Total Values:{total_values}")


def get_values(network):
    fc1_weight = net.state_dict()["fc1.weight"]
    fc1_bias = net.state_dict()["fc1.bias"]
    fc2_weight = net.state_dict()["fc2.weight"]
    fc2_bias = net.state_dict()["fc2.bias"]

    values = t.cat((fc1_weight.flatten(), fc1_bias, fc2_weight.flatten(), fc2_bias))

    print(values.numpy()[66780:66790])

    return values.numpy()


def load_values(network, values):
    load_dict = {}
    # fc1.weight: torch.Size([84, 784] 65856
    fc1_weight = values[0:65856]
    fc1_weight = t.tensor(np.reshape(np.array(fc1_weight), (84, 784)), dtype=t.float32)
    load_dict["fc1.weight"] = fc1_weight

    # fc1.bias torch.Size([84]) 65940
    fc1_bias = values[65856:65940]
    fc1_bias = t.tensor(np.array(fc1_bias), dtype=t.float32)
    load_dict["fc1.bias"] = fc1_bias

    # fc2.weight torch.Size([10, 84]) 66780
    fc2_weight = values[65940:66780]
    fc2_weight = t.tensor(np.reshape(np.array(fc2_weight), (10, 84)), dtype=t.float32)
    load_dict["fc2.weight"] = fc2_weight

    # fc2.bias torch.Size([10]) 66790
    fc2_bias = values[66780:66790]
    fc2_bias = t.tensor(np.array(fc2_bias), dtype=t.float32)
    load_dict["fc2.bias"] = fc2_bias

    network.load_state_dict(load_dict)


(
    train_images_flattened,
    train_one_hot_labels,
    train_labels,
    test_images_flattened,
    test_one_hot_labels,
    test_labels,
) = get_mnist_data_labels_neural()
mnist = MNISTData(train_images_flattened, train_one_hot_labels)
mnist_loader = DataLoader(mnist, batch_size=100, shuffle=True)

net = MNISTNet(
    training_data_loader=mnist_loader, test_data=test_images_flattened, test_labels=test_labels, epoch_count=5
)

start_accuracy = net.check_test_accuracy()
start_state = net.get_state()
print("++++++++++++++++++++++++++++++++++++++++")
for i in range(5):
    epoch_values, iteration_values = net.train()
    state = net.get_state()

print("++++++++++++++++++++++++++++++++++++++++")
start_state[66780:66790]
net.load_state(start_state)
end_accuracy = net.check_test_accuracy()

print(f"Start:{start_accuracy:0.4f} End:{end_accuracy:0.4f}")

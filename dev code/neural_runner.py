import argparse
import numpy as np
from neural_model import MNISTData, MNISTNet
from mnist_data_prep import get_mnist_data_labels_neural
from torch.utils.data import DataLoader
from charting_runner import neural_training_chart, save_run_info

algorithm_dict = {
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
    "rhc": "Random Hill Climbing",
    "backprop": "Gradient Descent Backpropagation",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Runner")

    parser.add_argument(
        "algorithm", choices=["rhc", "sa", "ga", "backprop"], help="The algorithm type: rhc, sa, ga, backprop"
    )
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-restart", type=int, help="Each restart value to be used for rhc")
    parser.add_argument("-temperature", type=float, help="Start temperature value to be used for sa")
    parser.add_argument("-min_temp", type=float, default=0.001, help="Minimum temperature value to be used for sa")
    parser.add_argument("-decay", type=str, help="Decay model type to be used for sa")
    parser.add_argument("-population", type=int, help="Each popuation value to be used for ga")
    parser.add_argument("-mutation", type=float, help="Each mutation value to be used for ga")
    parser.add_argument("-max_iters", type=int, default=20)
    parser.add_argument("-max_attempts", type=int, default=5)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-capture_iteration_values", type=int, default=0)
    args = parser.parse_args()

    algorithm_settings = vars(args)
    algorithm_settings["capture_iteration_values"] = (
        True if algorithm_settings["capture_iteration_values"] == 1 else False
    )

    print("Loading Data")
    (
        train_images_flattened,
        train_one_hot_labels,
        train_labels,
        test_images_flattened,
        test_one_hot_labels,
        test_labels,
    ) = get_mnist_data_labels_neural()

    print("Loading Network")
    mnist = MNISTData(train_images_flattened, train_one_hot_labels)
    mnist_loader = DataLoader(mnist, batch_size=100, shuffle=True)

    neural_net = MNISTNet(
        training_data_loader=mnist_loader,
        train_acc_data=train_images_flattened,
        train_acc_labels=train_labels,
        test_data=test_images_flattened,
        test_labels=test_labels,
        epoch_count=args.epochs,
    )

    # state = neural_net.get_state()
    # np.save("neural_state_1.npy", state)
    initial_state = np.load("neural_state.npy")
    neural_net.load_state(initial_state)

    print("Starting Training")
    training_time, epoch_values, iteration_values = neural_net.train_with_algorithm(
        algorithm_settings, capture_iteration_values=algorithm_settings["capture_iteration_values"]
    )

    print("Charting Results")
    epoch_values = np.array(epoch_values)
    neural_training_chart(
        epoch_values,
        title=f'{algorithm_dict[algorithm_settings["algorithm"]]}',
        sup_title=f"Loss Curve",
        chart_loss=True,
        algorithm_settings=algorithm_settings,
    )
    neural_training_chart(
        epoch_values,
        title=f'{algorithm_dict[algorithm_settings["algorithm"]]}',
        sup_title=f"Error Curve",
        chart_loss=False,
        algorithm_settings=algorithm_settings,
    )

    save_run_info(algorithm_settings, training_time, epoch_values)

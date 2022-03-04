import six
import sys

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

# import mlrose as mh
import numpy as np

from neural_model import MNISTData, MNISTNet
from mnist_data_prep import get_mnist_data_labels_neural
from torch.utils.data import DataLoader
from charting_runner import neural_training_chart

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

neural_net = MNISTNet(
    training_data_loader=mnist_loader,
    train_acc_data=train_images_flattened,
    train_acc_labels=train_labels,
    test_data=test_images_flattened,
    test_labels=test_labels,
    epoch_count=5,
)

epoch_values, iteration_values = neural_net.rhc_train(capture_iteration_values=False)
epoch_values = np.array(epoch_values)
neural_training_chart(
    epoch_values, start_node=0, title="RHC", sup_title="Z Loss Curve 100 Iterations", chart_loss=True
)
neural_training_chart(
    epoch_values, start_node=0, title="RHC", sup_title="Z Error Curve 100 Iterations", chart_loss=False
)


epoch_count = 150
neural_net = MNISTNet(
    training_data_loader=mnist_loader,
    train_acc_data=train_images_flattened,
    train_acc_labels=train_labels,
    test_data=test_images_flattened,
    test_labels=test_labels,
    epoch_count=epoch_count,
)
training_time, epoch_values, iteration_values = neural_net.train(capture_iteration_values=False)
epoch_values = np.array(epoch_values)
neural_training_chart(
    epoch_values,
    sup_title="Gradient Back Prop",
    title="Loss Curve",
    chart_loss=True,
    info_settings={"epoch_count": epoch_count},
)
neural_training_chart(
    epoch_values,
    sup_title="Gradient Back Prop",
    title="Error Curve",
    chart_loss=False,
    info_settings={"epoch_count": epoch_count},
)


def neural_model_train_loss(state, **kwargs):
    model = kwargs["model"]
    model.load_state(state)
    loss = model.get_state_loss()

    return loss


neural_fitness = mh.CustomFitness(neural_model_train_loss, problem_type="continuous", model=neural_net)

neural_problem = mh.ContinuousOpt(
    neural_net.state_length, neural_fitness, maximize=False, min_val=-10, max_val=10, step=0.1
)


old_acc = neural_net.check_test_accuracy()
old_loss = neural_net.get_state_loss()
print(f"Starting Acc:{old_acc:7.4f} Loss:{old_loss}")
for i in range(100):
    max_iters = 100
    max_attempts = 2
    best_state, best_fitness, learning_curve = mh.random_hill_climb(
        neural_problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        restarts=max_iters,
        init_state=best_state,
        curve=False,
    )

    neural_net.load_state(best_state)
    new_loss = neural_net.get_state_loss()
    new_acc = neural_net.check_test_accuracy()
    print(
        f"{i}: Acc:{new_acc:7.4f}  Acc Improvement:{new_acc - old_acc:10.6f} Loss improvement:{old_loss - best_fitness:10.6f} Check:{new_loss==best_fitness}"
    )
    neural_net.next_training_data()
    old_acc = new_acc
    old_loss = new_loss

import six
import sys

from sklearn.model_selection import learning_curve

sys.modules["sklearn.externals.six"] = six
import mlrose_hiive as mh

# import mlrose as mh
import numpy as np
from sklearn.metrics import accuracy_score

from mnist_data_prep import get_mnist_data_labels_neural

(
    train_images_flattened,
    train_one_hot_labels,
    train_labels,
    test_images_flattened,
    test_one_hot_labels,
    test_labels,
) = get_mnist_data_labels_neural()

nn_model1 = mh.NeuralNetwork(
    hidden_nodes=[84],
    activation="relu",
    algorithm="gradient_descent",
    max_iters=500,
    bias=True,
    is_classifier=True,
    learning_rate=0.0001,
    early_stopping=True,
    clip_max=5,
    max_attempts=100,
    random_state=3,
    curve=True,
)

nn_model1.fit(train_images_flattened, train_one_hot_labels)
nn_model1.fitness_curve()

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(train_images_flattened)

y_train_accuracy = accuracy_score(train_one_hot_labels, y_train_pred)


(
    x_train,
    y_train,
    train_labels,
    x_test,
    y_test,
    test_labels,
) = get_mnist_data_labels_neural()


grid_search_parameters = {
    "max_iters": [8, 16, 32],  # nn params
    "learning_rate": [0.001, 0.002],  # nn params
    "schedule": [mh.ArithDecay(1), mh.ArithDecay(100)],  # sa params
}

nnr = mh.NNGSRunner(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    experiment_name="nn_test",
    algorithm=mh.algorithms.sa.simulated_annealing,
    grid_search_parameters=grid_search_parameters,
    iteration_list=[1, 5, 10, 15, 20, 25, 30],
    hidden_layer_sizes=[[20]],
    bias=True,
    early_stopping=False,
    clip_max=1e10,
    max_attempts=500,
    generate_curves=True,
    seed=200972,
)

results = nnr.run()  # GridSearchCV instance returned

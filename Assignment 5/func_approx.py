from cgi import test
from json.tool import main
import sys 
sys.path.append("../")

from typing import Callable, Tuple, Sequence, Iterator, List
from scipy.stats import norm
import numpy as np
from rl.function_approx import LinearFunctionApprox, DNNApprox, AdamGradient, DNNSpec
from itertools import islice
from rl.gen_utils.plot_funcs import plot_list_of_curves
from matplotlib import pyplot as plt


def model_data_generator(SEED : int = 42) -> Iterator[Tuple[float, float]]:
    poly: Callable[[float], float] = lambda x : 3 * x ** 2 - 2 * x + 1
    trig: Callable[[float], float] = lambda x : 2 + np.sin(3 * np.pi * x)
    expo: Callable[[float], float] = lambda x : 2 * np.exp(x - 3)

    if SEED:
        np.random.seed(SEED)

    err = norm(loc = 0.0, scale = 0.75)

    while True:
        x = np.random.uniform(low = 0, high = 5)
        y = (poly(x) if x >= 0 and x <= 1 else trig(x) if x > 1 and x <= 3 else expo(x)) + err.rvs(size=1)[0]

        yield (x, y)

def data_seq_generator(
    data_generator: Iterator[Tuple[float, float]],
    num_pts: int
) -> Iterator[Sequence[Tuple[float, float]]]:
    while True:
        pts: Sequence[Tuple[float, float]] = list(islice(data_generator, num_pts))
        yield pts


def generate_features():
    return [lambda _ : 1.0, lambda x : x ** 2, lambda x : np.sin(x), lambda x : np.cos(x), lambda x : np.exp(x)]

def create_gradient():
    return AdamGradient(learning_rate=0.05, decay1=0.9, decay2=0.999)

def get_linear_model() -> LinearFunctionApprox[float]:
    feature_functions = generate_features()
    adam_grad = create_gradient()

    return LinearFunctionApprox.create(feature_functions=feature_functions, adam_gradient=adam_grad, regularization_coeff=0.0, direct_solve=True)

def get_dnn_model() -> DNNApprox[float]:
    feature_functions = generate_features()
    adam_grad = create_gradient()

    def relu(arg : np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x : x if x > 0. else 0.)(arg)

    def relu_deriv(arg : np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x : 1 if x > 0. else 0.)(arg)

    ds = DNNSpec(neurons=[4],
                 bias=True,
                 hidden_activation=relu,
                 hidden_activation_deriv=relu_deriv,
                 output_activation=relu,
                 output_activation_deriv=relu_deriv
                )

    return DNNApprox.create(feature_functions=feature_functions, dnn_spec=ds, adam_gradient=adam_grad, regularization_coeff=0.1)


if __name__ == "__main__":
    training_size: int = 500
    test_size: int = 120
    train_iterations: int = 100
    data_gen: Iterator[Tuple[float, float]] = model_data_generator()
    training_data_gen: Iterator[Sequence[Tuple[float, float]]] = data_seq_generator(data_gen, training_size)
    train_data: Sequence[Tuple[float, float]] = list(islice(data_gen, training_size))
    test_data: Sequence[Tuple[float, float]] = list(islice(data_gen, test_size))


    # direct_solve_lfa: LinearFunctionApprox[float] = get_linear_model().solve(next(training_data_gen))
    # direct_solve_rmse: float = direct_solve_lfa.rmse(test_data)
    # print(f"Linear Model Direct Solve RMSE = {direct_solve_rmse:.3f}")
    # print("-----------------------------")

    # print("Linear Model SGD")
    # print("----------------")
    # linear_model_rmse_seq: List[float] = []
    # for lfa in islice(get_linear_model().iterate_updates(training_data_gen), train_iterations):
    #     this_rmse: float = lfa.rmse(test_data)
    #     linear_model_rmse_seq.append(this_rmse)
    #     iter: int = len(linear_model_rmse_seq)
    #     print(f"Iteration {iter:d}: RMSE = {this_rmse:.3f}")

    # print("DNN Model SGD")
    # print("-------------")
    # dnn_model_rmse_seq: List[float] = []
    # for dfa in islice(get_dnn_model().iterate_updates(training_data_gen), train_iterations):
    #     this_rmse: float = dfa.rmse(test_data)
    #     dnn_model_rmse_seq.append(this_rmse)
    #     iter: int = len(dnn_model_rmse_seq)
    #     print(f"Iteration {iter:d}: RMSE = {this_rmse:.3f}")

    # x_vals = range(train_iterations)
    # plot_list_of_curves(
    #     list_of_x_vals=[x_vals, x_vals],
    #     list_of_y_vals=[linear_model_rmse_seq, dnn_model_rmse_seq],
    #     list_of_colors=["b", "r"],
    #     list_of_curve_labels=["Linear Model", "Deep Neural Network Model"],
    #     x_label="Iterations of Gradient Descent",
    #     y_label="Root Mean Square Error",
    #     title="RMSE across Iterations of Gradient Descent"
    # )

    dnn_model = get_dnn_model()

    for _ in range(train_iterations):
        dnn_model = dnn_model.update(train_data)
    
    int_1 = np.linspace(0, 1, 25)
    int_2 = np.linspace(1, 3, 50)
    int_3 = np.linspace(3, 5, 50)

    true_x = [i for i in int_1] + [i for i in int_2] + [i for i in int_3]
    true_y = [3 * x ** 2 - 2 * x + 1 for x in int_1] + [2 + np.sin(3 * np.pi * x) for x in int_2] + [2 * np.exp(x - 3) for x in int_3]

    # plt.plot(true_x, true_y, label = 'True function', color = 'red')
    # plt.scatter([x for x, _ in test_data], dnn_model.evaluate([x for x, _ in test_data]), label = 'Predicted', s = 15, color = 'green')
    # plt.scatter([x for x, _ in test_data], [y for _, y in test_data], color = 'black', s = 10)

    plt.plot(true_x, true_y, label = 'True function', color = 'red')
    plt.scatter([x for x, _ in train_data], dnn_model.evaluate([x for x, _ in train_data]), label = 'Predicted', s = 15, color = 'green')
    plt.scatter([x for x, _ in train_data], [y for _, y in train_data], color = 'black', s = 10)

    plt.legend()
    plt.show()


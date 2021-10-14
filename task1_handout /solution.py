import os
import typing

import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as ker
import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from time import time
import matplotlib.pyplot as plt
from copy import copy
from sklearn import svm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary=44):

    def preprocess(self, train_x, train_y, num_samples):
        """

            We subsample the dense region to create a balanced dataset and at the same time deal with
            the large scale of the dataset.

            Args:
                train_x (numpy.array):  training data points
                train_y (numpy.array):  training labels
                num_dense_labels (int): number of dense data points that should be subsampled
            Returns:
                train_x_s (numpy.array):    uniformly sampled and balanced training data
                train_y_s (numpy.array):    corresponding training labels
        """
        #

        data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
        
        np.random.shuffle(data)

        # random shuffle datapoints in the dense region and select 150 to balance the data
        train_x_sampled = data[:num_samples,:2]
        train_y_sampled = data[:num_samples,2]
        assert train_x_sampled.shape[0] == num_samples
        assert train_y_sampled.shape[0] == num_samples

        print("Sampled train_x shape:", train_x_sampled.shape)
        print("Sampled train_y shape:", train_y_sampled.shape)

        self.scaler = StandardScaler().fit(train_x_sampled)

        return train_x_sampled , train_y_sampled
        
    def predict(self, test_x):
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        # gp_mean = np.zeros(x.shape[0], dtype=float)
        # gp_std = np.zeros(x.shape[0], dtype=float)
        gp_mean,gp_stddev = self.model.predict(test_x, return_std=True)
        # TODO: Use the GP posterigor to form your predictions here
        # predict_safe = (gp_mean < THRESHOLD).astype(int)
        # y = gp_mean + predict_safe * 5

        # predict_safe = (gp_mean + 2* gp_stddev > THRESHOLD and THRESHOLD > gp_mean - 2* gp_stddev).astype(int)
        y = gp_mean
        np.where(np.logical_and(y + 2*gp_stddev >THRESHOLD , y - 2*gp_stddev <THRESHOLD),THRESHOLD,y)
        # np.where(y - 2*gp_stddev <THRESHOLD,THRESHOLD,y )
        # y = gp_mean - predict_safe*gp_mean + predict_safe*THRESHOLD
        return y,gp_mean,gp_stddev

    def fit_model(self, train_x, train_y):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        train_x, train_y = self.preprocess(train_x, train_y, 5000)
        k = ker.Matern(length_scale=0.01, nu=1.25) + \
            ker.WhiteKernel(noise_level=1e-7)

        gpr = gp.GaussianProcessRegressor(kernel=k, alpha=0.01, n_restarts_optimizer=5, random_state=42, normalize_y=True)
        noisyMat_gpr = pipeline.Pipeline([("scaler", self.scaler),("gpr", gpr)])

        print("Fitting noisy Matern GPR")
        start = time()
        noisyMat_gpr.fit(train_x, train_y)
        print("Took {} seconds".format(time() - start))
        self.model = noisyMat_gpr


def cost_function(y_true, y_predicted) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()

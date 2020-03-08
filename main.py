import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn_rvm import EMRVR
from skrvm import RVR


def generate_training_data(num_samples, noise_level, training_data_range):
    rng = np.random.RandomState(0)

    # Generate sample data
    X = training_data_range * (0.5 - rng.rand(num_samples, 1))
    y = np.sinc(X).ravel()
    # Add noise
    y += noise_level * 2 * (0.5 - rng.rand(X.shape[0]))

    return X, y


def get_plot_params():
    plot_params = dict()

    plot_params['x_high'] = 25
    plot_params['x_low'] = - plot_params['x_high']

    plot_params['y_high'] = 1.2
    plot_params['y_low'] = - 0.7

    return plot_params


def plot_results(X,
                 y,
                 rvr,
                 gpr,
                 X_plot,
                 y_rvr,
                 y_gpr,
                 rvr_name='RVR',
                 gpr_name='GPR',
                 y_rvr_std=None,
                 y_gpr_std=None,
                 noise_level=None,
                 training_data_range=None):
    plot_params = get_plot_params()

    # Plot results
    plt.figure(figsize=(15, 7))
    lw = 2

    plt.scatter(X, y, c='k', alpha=0.1, label='data')

    plt.plot(X_plot, np.sinc(X_plot), color='darkgreen', lw=lw, label='True')

    label_text = '{} ({})'.format(
        rvr_name,
        rvr.kernel
    )
    plt.plot(X_plot, y_rvr, color='navy', lw=lw, label=label_text)
    if y_rvr_std is not None:
        plt.fill_between(X_plot[:, 0], y_rvr - y_rvr_std, y_rvr + y_rvr_std, color='navy', alpha=0.2)

    label_text = '{} ({})'.format(
        gpr_name,
        gpr.kernel
    )
    plt.plot(X_plot, y_gpr, color='darkorange', lw=lw, label=label_text)
    if y_gpr_std is not None:
        plt.fill_between(X_plot[:, 0], y_gpr - y_gpr_std, y_gpr + y_gpr_std, color='darkorange', alpha=0.2)

    num_samples = len(y)

    plt.xlabel('data')
    plt.ylabel('target')
    plt.xlim(plot_params['x_low'], plot_params['x_high'])
    plt.ylim(plot_params['y_low'], plot_params['y_high'])
    if noise_level is None:
        if training_data_range is None:
            parenthesis_text = '{}={}'.format(
                '#training_samples',
                num_samples
            )
        else:
            parenthesis_text = '{}={}'.format(
                '#training_range',
                training_data_range
            )
    else:
        parenthesis_text = '{}={:.2}'.format(
            'noise_level',
            noise_level
        )
    title = '{} vs. {} ({})'.format(
        gpr_name,
        rvr_name,
        parenthesis_text,
    )
    plt.title(title)
    plt.legend(loc="best", scatterpoints=1, prop={'size': 8})
    plt.show()

    return


def benchmark():
    # Parameters to generate training data
    num_samples = 100
    noise_level = 0.1
    training_data_range = 10

    # Training data
    X, y = generate_training_data(num_samples, noise_level, training_data_range)

    # Fit
    gpr = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
    gpr.fit(X, y)

    rvr = RVR()
    rvr.fit(X, y)

    ## Implementation of RVR by sklearn_rvm
    emrvr = EMRVR(kernel='rbf',
                  gamma='scale')
    emrvr.fit(X, y)

    # Predict
    plot_params = get_plot_params()
    X_plot = np.linspace(plot_params['x_low'], plot_params['x_high'], 10000)[:, None]

    # Caveat:
    # generating the variance of the predictive distribution takes considerably longer than just predicting the mean.
    # Reference:
    # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html
    y_gpr, y_gpr_std = gpr.predict(X_plot, return_std=True)

    ## Implementation of RVR by skrvm
    y_rvr = rvr.predict(X_plot)
    y_rvr_std = None

    ## Implementation of RVR by sklearn_rvm
    y_emrvr, y_emrvr_std = emrvr.predict(X_plot, return_std=True)

    # Plot
    plot_results(X, y, rvr, gpr, X_plot, y_rvr, y_gpr,
                 "skrvm's RVR", "GPR", y_rvr_std, y_gpr_std,
                 training_data_range=training_data_range)

    plot_results(X, y, emrvr, gpr, X_plot, y_emrvr, y_gpr,
                 "sklearn_rvm's RVR", "GPR", y_emrvr_std, y_gpr_std,
                 training_data_range=training_data_range)

    plot_results(X, y, rvr, emrvr, X_plot, y_rvr, y_emrvr,
                 "skrvm's RVR", "sklearn_rvm's RVR", y_rvr_std, y_emrvr_std,
                 training_data_range=training_data_range)

    return


if __name__ == '__main__':
    benchmark()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn_rvm import EMRVR
from skrvm import RVR


def generate_training_data(num_samples, noise_level, training_data_range):
    rng = np.random.RandomState(0)

    # Generate sample data
    X = training_data_range * (0.5 - rng.rand(num_samples, 1))
    y = np.sinc(X).ravel()
    # Add noise
    # noinspection PyArgumentList
    y += noise_level * 2 * (0.5 - rng.rand(X.shape[0]))

    return X, y


def get_plot_params():
    plot_params = dict()

    plot_params['x_high'] = 25
    plot_params['x_low'] = - plot_params['x_high']

    plot_params['y_high'] = 1.2
    plot_params['y_low'] = - 0.7

    return plot_params


def plot_relevant_vectors(X,
                          y,
                          model,
                          model_name='RVR',
                          circle_size=80,
                          edgecolors='navy',
                          ax=None):
    try:
        # For any of the two implementations of RVR:
        model_relevance = model.relevance_
        # Caveat:
        # - for sklearn_rvm, these are indices of the relevant vectors
        # - for skrvm, these are actually abscissa of relevant vectors!
    except AttributeError:
        # GPR has no relevance vector:
        model_relevance = None

    if model_relevance is not None:
        try:
            # Implementation of RVR by sklearn_rvm
            relevance_vectors_idx = model_relevance
            X_relevant = X[relevance_vectors_idx]
            y_relevant = y[relevance_vectors_idx]
        except IndexError:
            # Implementation of RVR by skrvm: model.relevance_ is not the indices of relevant vectors, so we fix it now!
            X_relevant = model_relevance
            relevance_vectors_idx = [
                i for i in range(X.shape[0])
                if X[i] in X_relevant
            ]
            y_relevant = y[relevance_vectors_idx]

        if ax is None:
            scatter_fun = plt.scatter
        else:
            scatter_fun = ax.scatter

        scatter_fun(X_relevant,
                    y_relevant,
                    s=circle_size,
                    facecolors="none",
                    edgecolors=edgecolors,
                    label="relevance vectors ({})".format(model_name))

    return


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
                 rvr_color='navy',
                 gpr_color='darkorange',
                 ground_truth_color='darkgreen',
                 noise_level=None,
                 training_data_range=None,
                 ax=None):
    plot_params = get_plot_params()

    show_immediately = bool(ax is None)

    # Plot results
    if show_immediately:
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    lw = 2

    ax.scatter(X, y, c='k', alpha=0.1, label='data')

    ax.plot(X_plot, np.sinc(X_plot), color=ground_truth_color, lw=lw, label='True')

    label_text = '{} ({})'.format(
        rvr_name,
        rvr.kernel
    )
    ax.plot(X_plot, y_rvr, color=rvr_color, lw=lw, label=label_text)
    if y_rvr_std is not None:
        ax.fill_between(X_plot[:, 0], y_rvr - y_rvr_std, y_rvr + y_rvr_std, color=rvr_color, alpha=0.2)

    plot_relevant_vectors(X,
                          y,
                          model=rvr,
                          model_name=rvr_name,
                          circle_size=80,
                          edgecolors=rvr_color,
                          ax=ax)

    label_text = '{} ({})'.format(
        gpr_name,
        gpr.kernel
    )
    ax.plot(X_plot, y_gpr, color=gpr_color, lw=lw, label=label_text)
    if y_gpr_std is not None:
        ax.fill_between(X_plot[:, 0], y_gpr - y_gpr_std, y_gpr + y_gpr_std, color=gpr_color, alpha=0.2)

    # Circle size is different, so that we can see concentric circles if the relevant vectors are used by both models.
    plot_relevant_vectors(X,
                          y,
                          model=gpr,
                          model_name=gpr_name,
                          circle_size=50,
                          edgecolors=gpr_color,
                          ax=ax)

    num_samples = len(y)

    if show_immediately:
        ax.set_xlabel('data')
    ax.set_ylabel('target')
    ax.set_xlim(plot_params['x_low'], plot_params['x_high'])
    ax.set_ylim(plot_params['y_low'], plot_params['y_high'])
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
        rvr_name,
        gpr_name,
        parenthesis_text,
    )
    ax.set_title(title)
    ax.legend(loc="best", scatterpoints=1, prop={'size': 8})
    if show_immediately:
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

    ## Implementation of RVR by skrvm
    rvr = RVR(kernel='rbf')
    rvr.fit(X, y)

    ## Implementation of RVR by sklearn_rvm
    # Caveat: Since sklearn v.0.22, the default value of gamma changed from ‘auto’ to ‘scale’.
    # Reference: https://github.com/Mind-the-Pineapple/sklearn-rvm/issues/9
    emrvr = EMRVR(kernel='rbf',
                  gamma='auto')
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
    num_figures = 3
    fig, axs = plt.subplots(num_figures, 1, figsize=(15, 7))

    plot_results(X, y, rvr, gpr, X_plot, y_rvr, y_gpr,
                 "skrvm", "GPR", y_rvr_std, y_gpr_std,
                 rvr_color='purple', gpr_color='darkorange',
                 training_data_range=training_data_range,
                 ax=axs[0])

    plot_results(X, y, emrvr, rvr, X_plot, y_emrvr, y_rvr,
                 "sklearn_rvm", "skrvm", y_emrvr_std, y_rvr_std,
                 rvr_color='navy', gpr_color='purple',
                 training_data_range=training_data_range,
                 ax=axs[1])

    plot_results(X, y, emrvr, gpr, X_plot, y_emrvr, y_gpr,
                 "sklearn_rvm", "GPR", y_emrvr_std, y_gpr_std,
                 rvr_color='navy', gpr_color='darkorange',
                 training_data_range=training_data_range,
                 ax=axs[2])

    plt.show()

    return


if __name__ == '__main__':
    benchmark()

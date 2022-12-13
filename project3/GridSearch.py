import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from FFNN import FFNN
from ResampleMethods import *

cmt = 1 / 2.54


def GridSearch_FFNN_reg(
    X,
    y,
    n_layers,
    n_neurons,
    lambda_values,
    eta=0.001,
    plot_grid=True,
    gamma=0.9,
    activation_hidden='reLU',
    n_epochs=500,
    batch_size=20,
    k=5,
):

    rmse_values = np.zeros((len(n_layers), len(n_neurons), len(lambda_values)))
    r2_values = np.zeros((len(n_layers), len(n_neurons), len(lambda_values)))

    for i, L in enumerate(n_layers):
        for j, n in enumerate(n_neurons):
            for l, lmbda in enumerate(lambda_values):
                print(
                    f'Computing with {L} layers and {n} neurons and lambda={lmbda}.'
                )
                network = FFNN(
                    n_hidden_neurons=[n] * L,
                    task='regression',
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    eta=eta,
                    lmbda=lmbda,
                    gamma=gamma,
                    activation_hidden=activation_hidden,
                )

                mse, r2 = CrossValidation_regression(network, X, y, k=k)

                rmse_values[i][j][l] = np.sqrt(mse)
                r2_values[i][j][l] = r2

    if plot_grid:
        # make a 3d scatter plot with the MSE values as color
        fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.get_cmap('magma')

        # normalize the color to the range of the MSE values
        # normalize = matplotlib.colors.Normalize(vmin=np.min(rmse_values), vmax=np.max(rmse_values))
        for i, L in enumerate(n_layers):
            for j, n in enumerate(n_neurons):
                ax.scatter(
                    [L] * len(lambda_values),
                    [n] * len(lambda_values),
                    np.log10(lambda_values),
                    c=rmse_values[i][j],
                    vmin=np.min(rmse_values),
                    vmax=np.max(rmse_values),
                    cmap=cm,
                    s=100,
                )
                # add colorbar
        # put the colorbar on the bottom

        cbar = fig.colorbar(
            ax.collections[0],
            ax=ax,
            orientation='vertical',
            location='left',
            pad=0,
        )
        # normalize colorbar to the range of the MSE values
        cbar.mappable.set_clim(np.min(rmse_values), np.max(rmse_values))

        cbar.set_label('RMSE')

        ax.set_ylabel('Layers')
        ax.set_xlabel('Neurons')
        # put z lable to the left
        ax.set_zlabel('$\log_{10}(\lambda$)')
        # mark the minimum as an o and display mse value as text
        min_mse = np.min(rmse_values)
        min_mse_index = np.where(rmse_values == min_mse)
        ax.scatter(
            n_layers[min_mse_index[0][0]],
            n_neurons[min_mse_index[1][0]],
            np.log10(lambda_values[min_mse_index[2][0]]),
            s=120,
            marker='o',
            facecolors='none',
            edgecolors='r',
        )
        ax.text(
            n_layers[min_mse_index[0][0]],
            n_neurons[min_mse_index[1][0]],
            np.log10(lambda_values[min_mse_index[2][0]]),
            f'MSE={min_mse:.3f}',
            color='red',
            size=10,
        )

        info = (
            f'_mom{gamma}'
            + '_activ'
            + activation_hidden
            + f'_epoch{n_epochs}_batch{batch_size}_eta{eta}'
        )
        plt.tight_layout()
        # adjust tight layout to make room for colorbar
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2
        )
        plt.savefig('../figs/gridsearch_FFNN_reg_MSE' + info + '.pdf')

        # fig, ax = plt.subplots(figsize = (13*cmt, 12*cmt))
        # sns.heatmap(
        #    r2_values,
        #    annot=True,
        #    ax=ax,
        #    cmap="viridis",
        #    cbar_kws={'label': '$R^2$'},
        #    yticklabels=n_layers,
        #    xticklabels=n_neurons,
        #    zticklabels=np.round(np.log10(lambda_values), 2)
        #    )
        ##ax.set_title("$R^2$")
        # ax.set_ylabel("Layers")
        # ax.set_xlabel("Neurons")
        # ax.set_zlabel("$\log_{10}(\lambda$)")
        # plt.tight_layout()
        # plt.savefig("../figs/gridsearch_FFNN_reg_R2" + info + ".pdf")

    return (
        rmse_values,
        lambda_values[min_mse_index[2][0]],
        n_layers[min_mse_index[0][0]],
        n_neurons[min_mse_index[1][0]],
    )

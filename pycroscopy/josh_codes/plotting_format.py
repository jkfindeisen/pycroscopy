from sklearn import (decomposition)
import numpy as np
from scipy import (interpolate)
import matplotlib.pyplot as plt
import os

def conduct_PCA(loops, n_components=15, verbose=True):
    """
    Computes the PCA and forms a low-rank representation of a series of response curves
    This code can be applied to all forms of response curves.
    loops = [number of samples, response spectra for each sample]

    Parameters
    ----------
    loops : numpy array
        1 or 2d numpy array - [number of samples, response spectra for each sample]
    n_components : int, optional
        int - sets the number of componets to save
    verbose : bool, optional
        output operational comments

    Returns
    -------
    PCA : object
        results from the PCA
    PCA_reconstructed : numpy array
        low-rank representation of the raw data reconstructed based on PCA denoising
    """

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, f'shape of data resized to [{loops.shape[0]} x {loops.shape[1]}]')
    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    if np.isnan(loops).any():
        raise ValueError('data has infinite values consider using a imputer \n see interpolate_missing_points function')

    # Sets the number of components to save
    pca = decomposition.PCA(n_components=n_components)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA = pca.fit(loops)

    # does the inverse transform - creates a low rank representation of the data
    # this process denoises the data
    PCA_reconstructed = pca.inverse_transform(pca.transform(loops))

    # resized the array for hyperspectral data
    if loops.ndim == 3:
        PCA_reconstructed = PCA_reconstructed.reshape(original_size, original_size, -1)

    return(PCA, PCA_reconstructed)


def verbose_print(verbose, *args):
    if verbose:
        print(*args)

def interpolate_missing_points(loop_data):
    """
    Interpolates bad pixels in piezoelectric hystereis loops.\n
    The interpolation of missing points alows for machine learning operations

    Parameters
    ----------
    loop_data : numpy array
        arary of loops

    Returns
    -------
    loop_data_cleaned : numpy array
        arary of loops
    """

    # reshapes the data such that it can run with different data sizes
    if loop_data.ndim == 2:
        loop_data = loop_data.reshape(np.sqrt(loop_data.shape[0]),
                                                        np.sqrt(loop_data.shape[0]),-1)
        loop_data = np.expand_dims(loop_data, axis=0)
    elif loop_data.ndim == 3:
        loop_data = np.expand_dims(loop_data, axis=0)

    # Loops around the x index
    for i in range(loop_data.shape[0]):

        # Loops around the y index
        for j in range(loop_data.shape[1]):

            # Loops around the number of cycles
            for k in range(loop_data.shape[3]):

               if any(~np.isfinite(loop_data[i,j,:,k])):

                    true_ind = np.where(~np.isnan(loop_data[i,j,:,k]))
                    point_values = np.linspace(0,1,loop_data.shape[2])
                    spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                            loop_data[i,j,true_ind,k].squeeze())
                    ind = np.where(np.isnan(loop_data[i,j,:,k]))
                    val = spline(point_values[ind])
                    loop_data[i,j,ind,k] = val

    return loop_data.squeeze()

def layout_graphs_of_arb_number(graph):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    Parameters
    ----------
    graphs : int
        number of axes to make

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """

    # Selects the number of columns to have in the graph
    if graph < 3:
        mod = 2
    elif graph < 5:
        mod = 3
    elif graph < 10:
        mod = 4
    elif graph < 17:
        mod = 5
    elif graph < 26:
        mod = 6
    elif graph < 37:
        mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)

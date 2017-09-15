from sklearn import (decomposition)
import numpy as np
from scipy import (interpolate)
import matplotlib.pyplot as plt
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string

Path = path.Path
PathPatch = patches.PathPatch

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
        verbose_print(verbose, 'shape of data resized to [{0}x {1}]'.format(loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    if np.isnan(loops).any():
        raise ValueError(
            'data has infinite values consider using a imputer \n see interpolate_missing_points function')

    # Sets the number of components to save
    pca = decomposition.PCA(n_components=n_components)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA = pca.fit(loops)

    # does the inverse transform - creates a low rank representation of the data
    # this process denoises the data
    PCA_reconstructed = pca.inverse_transform(pca.transform(loops))

    # resized the array for hyperspectral data
    try:
        PCA_reconstructed = PCA_reconstructed.reshape(original_size, original_size, -1)
    except:
        pass

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
                                      np.sqrt(loop_data.shape[0]), -1)
        loop_data = np.expand_dims(loop_data, axis=0)
    elif loop_data.ndim == 3:
        loop_data = np.expand_dims(loop_data, axis=0)

    # Loops around the x index
    for i in range(loop_data.shape[0]):

        # Loops around the y index
        for j in range(loop_data.shape[1]):

            # Loops around the number of cycles
            for k in range(loop_data.shape[3]):

                if any(~np.isfinite(loop_data[i, j, :, k])):

                    true_ind = np.where(~np.isnan(loop_data[i, j, :, k]))
                    point_values = np.linspace(0, 1, loop_data.shape[2])
                    spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                      loop_data[i, j, true_ind, k].squeeze())
                    ind = np.where(np.isnan(loop_data[i, j, :, k]))
                    val = spline(point_values[ind])
                    loop_data[i, j, ind, k] = val

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


def plot_pca_maps(pca, loops, add_colorbars=True, verbose=False, letter_labels = False,
                                add_scalebar=False, filename='./PCA_maps', print_EPS=False,
                                print_PNG=False, dpi=300, num_of_plots = True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    pca : model
        previously computed pca
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    """
    if num_of_plots:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(loops.shape[0],loops.shape[1]))
    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    for i in range(num_of_plots):
        im = ax[i].imshow(pca.transform(loops)[:, i].reshape(original_size, original_size))
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')
        #

        if add_colorbars:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i+1), loc='bm')

        if add_scalebar is not False:
            add_scalebar_to_figure(ax[i], add_scalebar[0], add_scalebar[1])

    plt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)

def plot_pca_values(voltage, pca, num_of_plots = True, set_ylim=True, letter_labels = False,
                                filename='./PCA_vectors', print_EPS=False,
                                print_PNG=False, dpi=300):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    voltage : numpy array
        voltage vector for the hysteresis loop
    pca : model
        previously computed pca
    num_of_plots : int, optional
        number of principle componets to show
    set_ylim : int, optional
        optional manual overide of y scaler
    letter_labels : bool, optional
        adds letter labels for use in publications
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    """
    if num_of_plots:
        num_of_plots = pca.n_components_

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots)

    for i in range(num_of_plots):
        ax[i].plot(voltage, pca.components_[i], 'k')
        ax[i].set_xlabel('Voltage')
        ax[i].set_ylabel('Amplitude (Arb. U.)')
        ax[i].set_yticklabels('')
        #ax[i].set_title('PC {0}'.format(i+1))
        if not set_ylim:
            ax[i].set_ylim(set_ylim[0], set_ylim[1])

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='PC {0}'.format(i+1), loc='bm')

    ddplt.tight_layout(pad=0, h_pad=0)

    savefig(filename, dpi=300, print_EPS=print_EPS, print_PNG=print_PNG)

def add_colorbar(axes, plot, location='right', size=10, pad=0.05, format='%.1e'):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    axes : matplotlib plot
        Plot being references for the scalebar
    location : str, optional
        position to place the colorbar
    size : int, optional
        percent size of colorbar realitive to the plot
    pad : float, optional
        gap between colorbar and plot
    format : str, optional
        string format for the labels on colorbar
    """

    # Adds the scalebar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(location, size='{0}%'.format(size), pad=pad)
    cbar = plt.colorbar(plot, cax=cax, format=format)

# Function to add text labels to figure


def labelfigs(axes, number, style='wb', loc='br', string_add='', size=14, text_pos='center'):
    """
    Adds labels to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    number : int
        letter number
    style : str, optional
        sets the color of the letters
    loc : str, optional
        sets the location of the label
    string_add : str, optional
        custom string as the label
    size : int, optional
        sets the fontsize for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formating_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formating_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError('Unknown string format imported please look at code for acceptable positions')

    if string_add == '':

        # Turns to image number into a label
        if number < 26:
            axes.text(x_value, y_value, string.ascii_lowercase[number],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axes.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:

        axes.text(x_value, y_value, string_add,
                  size=14, weight='bold', ha=text_pos,
                  va='center', color=formatting['color'],
                  path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                       foreground="k")])


def add_scalebar_to_figure(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds scalebar to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(np.floor(x_lim[1] - x_lim[0])), np.abs(np.floor(y_lim[1] - y_lim[0]))

    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.floor(image_size))
    y_point = np.linspace(y_lim[0], y_lim[1], np.floor(image_size))

    if loc == 'br':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.1 * image_size // 1)]
        y_end = y_point[np.int((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.9 * image_size // 1)]
        y_end = y_point[np.int((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int((.9 - .075) * image_size // 1)]

    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])

def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,ls=linestyle,lw=lineweight)
    axes.add_patch( pathpatch )

def savefig(filename, dpi=300, print_EPS=False, print_PNG = False):
    """
    Adds path to figure

    Parameters
    ----------
    filename : str
        path to save file
    dpi : int, optional
        resolution to save image
    print_EPS : bool, optional
        selects if export the EPS
    print_PNG : bool, optional
        selects if print the PNG
    """
    # Saves figures at EPS
    if print_EPS:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=dpi, bbox_inches='tight')

    # Saves figures as PNG
    if print_PNG:
        plt.savefig(filename + '.png', format='png',
                    dpi=dpi, bbox_inches='tight')

def pca_weights_as_embeddings(a=0):
    #TODO add function that takes PCA compoents and converts them to embeddings
    print('todo')

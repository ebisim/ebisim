"""
All the plotting logic of ebisim is collected in this module. The functions can be called manually
by the user, but are primarily desinged to be called internally by ebisim, thefore the API may lack
convenience in some places.
"""

from datetime import datetime
from math import atan2, degrees
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from . import xs
from .elements import Element


#: The default colormap used to grade line plots, assigning another colormap to this object
#: will result in an alternative color gradient for line plots
COLORMAP = plt.cm.plasma #pylint: disable=E1101


########################
#### E scan Plotting ###
########################

def plot_energy_scan(energies, abundance, cs=None, **kwargs):
    """
    Produces a plot of the charge state abundance for different energies at a given time.

    Parameters
    ----------
    energies : numpy.array
        <eV>
        The evaluated energies.
    abundance : numpy.array
        The abundance of each charge state (rows) for each energy (columns).
    cs : list of int or None, optional
        If None, all charge states are plotted. By supplying a list of int it
        is possible to filter the charge states that should be plotted.
        By default None.
    **kwargs
        Keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n = abundance.shape[0]
    _set_line_prop_cycle(ax, n)

    for c in range(n):
        if cs is None or c in cs:
            ax.plot(energies, abundance[c, :], figure=fig, label=f"{c}+")
        else:
            ax.plot([], [], figure=fig)

    kwargs.setdefault("xlim", (energies.min(), energies.max()))
    kwargs.setdefault("xlabel", "Electron energy (eV)")
    kwargs.setdefault("ylabel", "Abundance")
    decorate_axes(ax, **kwargs)

    return fig


def plot_energy_time_scan(energies, times, abundance, **kwargs):
    """
    Provides information about the abundance of a single charge states at all simulated times
    and energies.

    Parameters
    ----------
    energies : numpy.array
        <eV>
        The evaluated energies.
    times : numpy.array
        <s>
        The evaluated timesteps.
    abundance : numpy.array
        Abundance of charge state 'cs' at given times (rows) and energies (columns).
    **kwargs
        Keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    e_kin, t = np.meshgrid(energies, times)
    levels = np.arange(21)/20 * abundance.max()

    plot = ax.contourf(e_kin, t, abundance, levels=levels, cmap="plasma")
    plt.colorbar(plot, ticks=np.arange(0, 1.1, 0.1) * abundance.max())
    ax.contour(e_kin, t, abundance, levels=levels, colors="k", linewidths=.5)

    kwargs.setdefault("xlabel", "Electron energy (eV)")
    kwargs.setdefault("ylabel", "Time (s)")
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("grid", False)
    kwargs["label_lines"] = False
    decorate_axes(ax, **kwargs)

    return fig


###########################
#### Evolution Plotting ###
###########################

def plot_generic_evolution(t, y, plot_total=False, ax=None, **kwargs):
    """
    Plots the evolution of a quantity with time

    Parameters
    ----------
    t : numpy.array
        <s>
        Values for the time steps.
    y : numpy.array
        Values of the evoloving quantity to plot as a function of time.
        Has to be a 2D numpy array where the rows correspond to the different charge states and
        the columns correspond to the individual timesteps.
    plot_total : bool, optional
        Indicate whether a black dashed line indicating the total accross all charge states should
        also be plotted, by default False.
    ax : matplotlib.Axes, optional
        Provide if you want to add this plot to existing Axes, by default None.
    **kwargs
        Keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.get_figure()

    n = y.shape[0]
    _set_line_prop_cycle(ax, n)

    ls = kwargs.pop("ls", None)
    for cs in range(n):
        if np.array_equal(np.unique(y[cs, :]), np.array([0])):
            ax.loglog([], []) # Ghost draw for purely zero cases
        else:
            if ls:
                ax.loglog(t, y[cs, :], ls=ls, label=str(cs) + "+")
            else:
                ax.loglog(t, y[cs, :], label=str(cs) + "+")
    if plot_total:
        ax.plot(t, np.sum(y, axis=0), c="k", ls="--", figure=fig, label="total")

    kwargs.setdefault("xlim", (1e-4, 1e3))
    kwargs.setdefault("xlabel", "Time (s)")
    kwargs.setdefault("ylabel", "Abundance")
    kwargs.setdefault("xscale", "log")
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("legend", False)
    kwargs.setdefault("label_lines", True)
    decorate_axes(ax, **kwargs)

    return fig


########################
#### Radial Plotting ###
########################

def plot_radial_distribution(r, dens, phi=None, r_e=None, ax=None, ax2=None, **kwargs):
    """
    Plots the radial ion distribution, can also plot radial potential and electron beam radius.

    Parameters
    ----------
    r : numpy.ndarray
        [description]
    dens : numpy.ndarray
        Array of densities, shaped like 'y' in plot_generic_evolution.
    phi : numpy.ndarray, optional
        The radial potential, if supplied will be plotted on second y-axis, by default None.
    r_e : numpy.ndarray, optional
        Electron beam radius, if provided will be marked as vertical likne, by default None.
    ax : numpy.ndarray, optional
        Axes on which to plot the densities, by default None.
    ax2 : numpy.ndarray, optional
        Axes on which to plot the radial potential, by default None.

    Returns
    -------
    ax : matplotlib.Axes
        As above.
    ax2 : matplotlib.Axes
        As above.
    """
    if ax is None:
        _, ax = plt.subplots()
    _ = ax.get_figure()

    kwargs.setdefault("xlabel", "Radius (m)")
    kwargs.setdefault("ylabel", "Density (m$^{-3}$)")
    ylimphi = kwargs.pop("ylimphi", None)
    plot_generic_evolution(r, dens, plot_total=True, ax=ax, **kwargs)
    if r_e is not None:
        ax.axvline(r_e, c="k", ls="--")
    if phi is not None:
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.set(ylabel="Radial potential (V)", yscale="linear")
        ax2.plot(r, phi, "k")
        if ylimphi:
            ax2.set_ylim(ylimphi)
    plt.tight_layout()
    return ax, ax2

########################
#### XS Plotting #######
########################

def _plot_xs(e_samp, xs_scan, ax=None, ls="-", **kwargs):
    """
    Low level plotting routine serving plot_eixs, plot_rrxs and plot_drxs

    Parameters
    ----------
    e_samp : numpy.ndarray
        <eV>
        Array holding the sampling energies.
    xs_scan : numpy.ndarray
        <m^2>
        Array holding the cross sections, where the row index corresponds to the charge state
        and the columns correspond to the different sampling energies.
    ax : matplotlib.Axes, optional
        Provide if you want to add this plot to existing Axes, by default None.
    ls : str, optional
        Can be provided to define a linestyle for the plot, by default "-".
    **kwargs
        Keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n = xs_scan.shape[0]

    ax.set_prop_cycle(None) # Reset property (color) cycle, needed when plotting on existing fig
    _set_line_prop_cycle(ax, n)

    for cs in range(n):
        xs_cs = xs_scan[cs, :]
        if np.array_equal(np.unique(xs_cs), np.array([0])):
            plt.plot([], []) # If all xs are zero, do a ghost plot to advance color cycle
        else:
            plt.plot(e_samp, 1e4*xs_cs, figure=fig, ls=ls, label=str(cs)+"+") # otherwise plot data

    kwargs.setdefault("xlim", (e_samp[0], e_samp[-1]))
    kwargs.setdefault("xscale", "log")
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("xlabel", "Electron kinetic energy (eV)")
    kwargs.setdefault("ylabel", "Cross section ($\\mathsf{cm^2}$)")
    kwargs.setdefault("legend", False)
    kwargs.setdefault("label_lines", True)
    kwargs.setdefault("grid", True)
    decorate_axes(ax, **kwargs)

    return fig


def plot_eixs(element, **kwargs):
    """
    Creates a figure showing the electron ionisation cross sections of the provided element.

    Parameters
    ----------
    element : ebisim.elements.Element or str or int
        An instance of the Element class, or an identifier for the element, i.e. either its
        name, symbol or proton number.
    **kwargs
        'fig' is intercepted and can be used to plot on top of an existing figure.
        'ls' is intercepted and can be used to set the linestyle for plotting.
        Remaining keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    element = Element.as_element(element)

    e_samp, xs_scan = xs.eixs_energyscan(element)

    kwargs.setdefault("title", f"{element.latex_isotope()} - EI")

    return _plot_xs(e_samp, xs_scan, **kwargs)


def plot_rrxs(element, **kwargs):
    """
    Creates a figure showing the radiative recombination cross sections of the provided element.

    Parameters
    ----------
    element : ebisim.elements.Element or str or int
        An instance of the Element class, or an identifier for the element, i.e. either its
        name, symbol or proton number.
    **kwargs
        'fig' is intercepted and can be used to plot on top of an existing figure.
        'ls' is intercepted and can be used to set the linestyle for plotting.
        Remaining keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    element = Element.as_element(element)

    e_samp, xs_scan = xs.rrxs_energyscan(element)

    kwargs.setdefault("title", f"{element.latex_isotope()} - RR")

    return _plot_xs(e_samp, xs_scan, **kwargs)


def plot_drxs(element, fwhm, **kwargs):
    """
    Creates a figure showing the dielectronic recombination cross sections of the provided element.

    Parameters
    ----------
    element : ebisim.elements.Element or str or int
        An instance of the Element class, or an identifier for the element, i.e. either its
        name, symbol or proton number.
    fwhm : float
        <eV>
        Energy spread to apply for the resonance smearing, expressed in terms of
        full width at half maximum.
    **kwargs
        'fig' is intercepted and can be used to plot on top of an existing figure.
        'ls' is intercepted and can be used to set the linestyle for plotting.
        Remaining keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    element = Element.as_element(element)

    e_samp, xs_scan = xs.drxs_energyscan(element, fwhm)

    kwargs.setdefault("xscale", "linear")
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("legend", True)
    kwargs.setdefault("label_lines", False)
    kwargs.setdefault(
        "title",
        f"{element.latex_isotope()} - DR (FWHM = {fwhm:0.1f} eV)"
    )

    return _plot_xs(e_samp, xs_scan, **kwargs)


def plot_combined_xs(element, fwhm, **kwargs):
    """
    Creates a figure showing the electron ionisation, radiative recombination and,
    dielectronic recombination cross sections of the provided element.

    Parameters
    ----------
    element : ebisim.elements.Element or str or int
        An instance of the Element class, or an identifier for the element, i.e. either its
        name, symbol or proton number.
    fwhm : float
        <eV>
        Energy spread to apply for the resonance smearing, expressed in terms of
        full width at half maximum.
    **kwargs
        Remaining keyword arguments are handed down to ebisim.plotting.decorate_axes,
        cf. documentation thereof.
        If no arguments are provided, reasonable default values are injected.

    Returns
    -------
    matplotlib.Figure
        Figure handle of the generated plot.
    """
    element = Element.as_element(element)

    kwargs.setdefault("xscale", "linear")
    kwargs.setdefault("yscale", "log")
    kwargs.setdefault("ylim", (1e-24, 1e-16))
    kwargs.setdefault("legend", True)
    kwargs.setdefault(
        "title",
        f"{element.latex_isotope()} - EI / RR / DR " \
        f"(FWHM = {fwhm:0.1f} eV)"
    )
    kwargs.setdefault("label_lines", True)

    label_lines = kwargs.pop("label_lines")
    legend = kwargs.pop("legend")

    fig = plot_eixs(
        element,
        ls="--",
        legend=legend,
        label_lines=False,
        **kwargs
    )
    fig = plot_rrxs(
        element,
        ax=fig.gca(),
        ls="-.",
        label_lines=label_lines,
        legend=False,
        **kwargs
    )
    fig = plot_drxs(
        element,
        fwhm,
        ax=fig.gca(),
        ls="-",
        legend=False,
        **kwargs
    )

    return fig


########################
#### Helper Methods ####
########################

def _set_line_prop_cycle(ax, n_lines):
    color = [COLORMAP(i) for i in np.linspace(0, .9, n_lines)]
    lw = [.75 if (i % 5 != 0) else 1.5 for i in range(n_lines)]
    ls = ["-" if (i % 5 != 0) else "-." for i in range(n_lines)]
    ax.set_prop_cycle(color=color, linewidth=lw, linestyle=ls)


def decorate_axes(ax, grid=True, legend=False, label_lines=True, tight_layout=True, **kwargs):
    """
    This function exists to have a common routine for setting certain figure properties, it is
    called by all other plotting routines and takes over the majority of the visual polishing.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to be modifed.
    grid : bool, optional
        Whether or not to lay a grid over the plot, by default True.
    legend : bool, optional
        Whether or not to put a legend next to the plot, by default False.
    label_lines : bool, optional
        Whether or not to put labels along the lines in the plot, by default True.
    tight_layout : bool, optional
        Whether or not to apply matplotlibs tight layout on the parentfigure of ax, by default True.
    **kwargs
        Are directly applied as axes properties, e.g. xlabel, xscale, title, etc.
    """
    ax.set(**kwargs)
    if grid:
        ax.grid(which="both", alpha=0.5, lw=0.5)
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Label lines should be called at the end of the plot generation since it relies on axlim
    if label_lines:
        # TODO: Check if this can be done without private member access
        lines = [l for l in ax.get_lines() if any(l._x)] # pylint: disable=W0212
        step = int(np.ceil(len(lines)/10))
        lines = lines [::step]
        _labelLines(lines, size=7, bbox={"pad":0.1, "fc":"w", "ec":"none"})
    if tight_layout:
        ax.figure.tight_layout()

###########################################################################################
#### Code for decorating line plots with online labels
#### Code copied from https://github.com/cphyc/matplotlib-label-lines
#### Based on https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
###########################################################################################
# Label line with line2D label data
def _labelLine(line, x, label=None, align=True, **kwargs):
    '''Label a single matplotlib line at position x'''
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # Convert datetime objects to floats
    if isinstance(x, datetime):
        x = date2num(x)

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the
    ip = 1
    for i, xd in enumerate(xdata):
        if x < xd:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1]) * \
        (x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if any(np.isnan([x, y])) or any(np.isinf([x,y])):
        return

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang, )), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def _labelLines(lines, align=True, xvals=None, **kwargs):
    '''Label all lines with their respective legends.

    xvals: (xfirst, xlast) or array of position. If a tuple is provided, the
    labels will be located between xfirst and xlast (in the axis units)

    '''
    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xvals = ax.get_xlim() # set axis limits as annotation limits, xvals now a tuple
    if isinstance(xvals, tuple):
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        _labelLine(line, x, label, align, **kwargs)

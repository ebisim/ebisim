"""
Module containing logic for plotting
"""
from datetime import datetime
from math import atan2, degrees
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from . import xs
from . import elements

COLORMAP = plt.cm.gist_rainbow


########################
#### E scan Plotting ###
########################


def plot_energy_scan(data, cs, ylim=None, title=None, invert_hor=False, x2fun=None, x2label=""):
    """
    Plots the charge state abundance vs the energy
    """
    fig = plt.figure(figsize=(6, 3), dpi=150)
    ax1 = fig.add_subplot(111)
    n = data.shape[1] - 1
    _set_line_prop_cycle(ax1, n)

    for c in range(n):
        if c in cs:
            plt.plot(data["e_kin"], data[c], figure=fig, label=str(c) + "+")
        else:
            plt.plot([], [], figure=fig)


    xlim = (data["e_kin"].min(), data["e_kin"].max())
    xlabel = "Electron Energy (eV)"
    ylabel = "Abundance"
    _decorate_axes(ax1, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    if invert_hor:
        plt.gca().invert_xaxis()

    if x2fun:
        ax2 = ax1.twiny()
        def tick_function(x):
            """tick function for the second axis"""
            V = x2fun(x)
            return ["%.0f" % z for z in V]
        new_tick_locations = ax1.get_xticks()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(x2label)
        if title:
            title += "\n\n"
    if title:
        ax1.set_title(title)
    plt.tight_layout()

    return fig

def plot_energy_time_scan(data, cs, xlim=None, ylim=None, title=None):
    """
    Plots the abundance of a charge state depending on the breeding time and energy
    """
    plotdf = data[["t", "e_kin", cs]].rename(columns={cs:"DATA"})
    nt = len(plotdf.t.unique())
    e_kin = plotdf.e_kin.values.reshape((-1, nt))
    t = plotdf.t.values.reshape((-1, nt))
    abd = plotdf.DATA.values.reshape((-1, nt))

    fig = plt.figure(figsize=(6, 3), dpi=150)
    ax = fig.add_subplot(111)

    levels = np.arange(21)/20
    plot = ax.contourf(e_kin, t, abd, levels=levels, cmap="plasma")
    ax.set_yscale("log")
    plt.colorbar(plot, ticks=np.arange(0, 1.1, 0.1))
    ax.contour(e_kin, t, abd, levels=levels, colors="k", linewidths=.5)
    _decorate_axes(ax, title=title,
                   xlabel="Electron kinetic energy (eV)", ylabel="Time (s)",
                   xlim=xlim, ylim=ylim, grid=False, legend=False, label_lines=False)
    return fig


########################
#### CS Evo Plotting ###
########################


def plot_cs_evolution(ode_solution, xlim=(1e-4, 1e3), ylim=(1e-4, 1),
                      title="Charge state evolution", legend=False, label_lines=True):
    """
    Method that plots the solution of an EBIS charge breeding simulation
    returns figure handle

    ode_solution - solution object to plot
    title - (optional) Plot title
    xlim, ylim - (optional) plot limits
    legend - (optional) show legend?
    line_labels - annotate lines?
    """
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)

    n = ode_solution.y.shape[0]
    _set_line_prop_cycle(ax, n)

    for cs in range(n):
        if np.array_equal(np.unique(ode_solution.y[cs, :]), np.array([0])):
            plt.semilogx([], [], figure=fig) # Ghost draw for purely zero cases
        else:
            plt.semilogx(ode_solution.t, ode_solution.y[cs, :], figure=fig, label=str(cs) + "+")

    _decorate_axes(ax, title=title, xlabel="Time (s)", ylabel="Relative Abundance",
                   xlim=xlim, ylim=ylim, grid=True, legend=legend, label_lines=label_lines)
    return fig

def plot_generic_evolution(t, y, xlim=(1e-4, 1e3), ylim=None, ylabel="", title="",
                           xscale="log", yscale="log", legend=False, label_lines=True,
                           plot_total=False):
    """
    Method that plots the evolution of a quantity of an EBIS charge breeding simulation
    returns figure handle

    ode_solution - solution object to plot
    title - (optional) Plot title
    xlim, ylim - (optional) plot limits
    ylabel - label
    legend - (optional) show legend?
    line_labels - annotate lines?
    """
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)

    n = y.shape[0]
    _set_line_prop_cycle(ax, n)

    for cs in range(n):
        if np.array_equal(np.unique(y[cs, :]), np.array([0])):
            plt.loglog([], [], figure=fig) # Ghost draw for purely zero cases
        else:
            plt.loglog(t, y[cs, :], figure=fig, label=str(cs) + "+")
    if plot_total:
        plt.plot(t, np.sum(y, axis=0), c="k", ls="--", figure=fig, label="total")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    _decorate_axes(ax, title=title, xlabel="Time (s)", ylabel=ylabel,
                   xlim=xlim, ylim=ylim, grid=True, legend=legend, label_lines=label_lines)
    return fig

########################
#### XS Plotting #######
########################

def _plot_xs(e_samp, xs_scan, fig=None, xscale="log", yscale="log",
             title=None, xlim=None, ylim=None, legend=False, label_lines=True,
             ls="-"):
    """
    Creates a figure showing the cross sections and returns the figure handle

    Input Parameters
    xs_df - dataframe holding the required data, one column must be ekin (energy) other columns
            represent each chargestate (columns should be in ascending order)
    fig - (optional) Pass handle to plot on existing figure
    xscale, yscale - (optional) Scaling of x and y axis (log or linear)
    title - (optional) Plot title
    xlim, ylim - (optional) plot limits
    legend - (optional) show legend?
    line_labels - annotate lines?
    ls - linestyle
    """
    n_cs = xs_scan.shape[0]

    if not fig:
        fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.gca()

    ax.set_prop_cycle(None) # Reset property (color) cycle, needed when plotting on existing fig
    _set_line_prop_cycle(ax, n_cs)

    for cs in range(n_cs):
        xs_cs = xs_scan[cs, :]
        if np.array_equal(np.unique(xs_cs), np.array([0])):
            plt.plot([], []) # If all xs are zero, do a ghost plot to advance color cycle
        else:
            plt.plot(e_samp, 1e4*xs_cs, figure=fig, ls=ls, label=str(cs)+"+") # otherwise plot data

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if not xlim:
        xlim = (e_samp[0], e_samp[-1])
    _decorate_axes(ax, title=title,
                   xlabel="Electron kinetic energy (eV)", ylabel="Cross section (cm$^2$)",
                   xlim=xlim, ylim=ylim, grid=True, legend=legend, label_lines=label_lines)
    return fig


def plot_ei_xs(element, **kwargs):
    """
    Creates a figure showing the EI cross sections and returns the figure handle

    Input Parameters
    **kwargs - passed on to _plot_xs, check arguments thereof
    """
    if not isinstance(element, elements.Element):
        element = elements.Element(element)

    e_samp, xs_scan = xs.eixs_energyscan(element)

    if kwargs.get("title") is None:
        kwargs["title"] = f"EI cross sections of {element.latex_isotope()}"

    return _plot_xs(e_samp, xs_scan, **kwargs)


def plot_rr_xs(element, **kwargs):
    """
    Creates a figure showing the RR cross sections and returns the figure handle

    Input Parameters
    **kwargs - passed on to _plot_xs, check arguments thereof
    """
    if not isinstance(element, elements.Element):
        element = elements.Element(element)

    e_samp, xs_scan = xs.rrxs_energyscan(element)

    if kwargs.get("title") is None:
        kwargs["title"] = f"RR cross sections of {element.latex_isotope()}"

    return _plot_xs(e_samp, xs_scan, **kwargs)


def plot_dr_xs(element, fwhm, **kwargs):
    """
    Creates a figure showing the DR cross sections and returns the figure handle

    Input Parameters
    **kwargs - passed on to _plot_xs, check arguments thereof
    """
    if not isinstance(element, elements.Element):
        element = elements.Element(element)
    e_samp, xs_scan = xs.drxs_energyscan(element, fwhm)
    # Set some kwargs if they are not given by caller
    kwargs["xscale"] = kwargs.get("xscale", "linear")
    kwargs["yscale"] = kwargs.get("yscale", "linear")
    kwargs["legend"] = kwargs.get("legend", True)
    kwargs["label_lines"] = kwargs.get("label_lines", False)
    # call _plot_xs with correct title and data
    if kwargs.get("title") is None:
        kwargs["title"] = f"DR cross sections of {element.latex_isotope()} " \
                          f"(Electron beam FWHM = {fwhm:0.1f} eV)"
    fig = _plot_xs(e_samp, xs_scan, **kwargs)
    # Return figure handle
    return fig

def plot_combined_xs(element, fwhm, xlim=None, ylim=(1e-24, 1e-16),
                     xscale="linear", yscale="log", legend=True):
    """
    Combo Plot of EI RR DR for element with fwhm for DR resonances
    """
    if not isinstance(element, elements.Element):
        element = elements.Element(element)
    title = f"Combined cross sections of {element.latex_isotope()} " \
            f"(Electron beam FWHM = {fwhm:0.1f} eV)"
    common_kwargs = dict(xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)
    fig = plot_ei_xs(element, label_lines=False, legend=legend, ls="--", **common_kwargs)
    fig = plot_rr_xs(element, fig=fig, ls="-.", **common_kwargs)
    fig = plot_dr_xs(element, fwhm, fig=fig, ls="-", legend=False, title=title, **common_kwargs)
    return fig


########################
#### Helper Methods ####
########################

def _set_line_prop_cycle(ax, n_lines):
    color = [COLORMAP(i) for i in np.linspace(0, 1, n_lines)]
    # color = [COLORMAP((i%10)/10) for i in range(n_lines)]
    lw = [.75 if (i % 5 != 0) else 1.5 for i in range(n_lines)]
    ax.set_prop_cycle(color=color, linewidth=lw)


def _decorate_axes(ax, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True,
                   legend=False, label_lines=True, tight_layout=True):
    """
    helper functions for common axes decorations
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(which="both", alpha=0.5, lw=0.5)
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Label lines should be called at the end of the plot generation since it relies on axlim
    if label_lines:
        lines = [l for l in ax.get_lines() if any(l._x)]
        step = int(np.ceil(len(lines)/10))
        lines = lines [::step]
        labelLines(lines, size=7, bbox={"pad":0.1, "fc":"w", "ec":"none"})
    if tight_layout:
        ax.figure.tight_layout()

###########################################################################################
#### Code for decorating line plots with online labels
#### Code pulled from https://github.com/cphyc/matplotlib-label-lines
#### Based on https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
###########################################################################################
# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):
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


def labelLines(lines, align=True, xvals=None, **kwargs):
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
        labelLine(line, x, label, align, **kwargs)

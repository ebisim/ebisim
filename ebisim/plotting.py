"""
Module containing logic for plotting
"""
from datetime import datetime
from math import atan2, degrees
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def plot_cs_evolution(ode_solution, xlim=(1e-4, 1e3), ylim=(1e-4, 1),
                      title="Charge State Evolution", legend=False, label_lines=True):
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
    ax = fig.gca()

    for cs in range(ode_solution.y.shape[0]):
        plt.semilogx(ode_solution.t, ode_solution.y[cs, :], figure=fig, label=str(cs) + "+")

    _decorate_axes(ax, title=title, xlabel="Time (s)", ylabel="Relative Abundance",
                   xlim=xlim, ylim=ylim, grid=True, legend=legend, label_lines=label_lines)
    return fig

def plot_xs(xs_df, fig=None, xscale="log", yscale="log",
            title=None, xlim=None, ylim=None, legend=False, label_lines=True):
    """
    Creates a figure showing the cross sections and returns the figure handle

    Input Parameters
    xs_df - dataframe holding the required data, one column must be ekin (energy) other columns
            represent each chargestate (columns should be in ascending order)
    fig - (optional) Pass hangle to lot on existing figure
    xscale, yscale - (optional) Scaling of x and y axis (log or linear)
    title - (optional) Plot title
    xlim, ylim - (optional) plot limits
    legend - (optional) show legend?
    line_labels - annotate lines?
    """
    if not fig: fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.gca()

    ekin = xs_df.ekin
    xs_df = xs_df.drop("ekin", axis=1)
    ax.set_prop_cycle(None) # Reset property (color) cycle, needed when plotting on existing fig
    for (cs, xs) in xs_df.iteritems():
        if np.array_equal(xs.unique(), np.array([0])):
            plt.plot([], []) # If all xs are zero, do a ghost plot to advance color cycle
        else:
            plt.plot(ekin, xs, figure=fig, lw=1, label=str(cs)+"+") # otherwise plot data

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _decorate_axes(ax, title=title,
                   xlabel="Electron kinetic energy (eV)", ylabel="Cross section (cm$^2$)",
                   xlim=xlim, ylim=ylim, grid=True, legend=legend, label_lines=label_lines)
    return fig

def _decorate_axes(ax, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True,
                   legend=False, label_lines=True):
    """
    helper functions for common axes decorations
    """
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if grid: ax.grid(which="both", alpha=0.5)
    if legend: ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Label lines should be called at the end of the plot generation since it relies on axlim
    if label_lines: labelLines(ax.get_lines(), size=7, bbox={"pad":0.1, "fc":"w", "ec":"none"})

####
#### Code for decorating line plots with online labels
#### Code pulled from https://github.com/cphyc/matplotlib-label-lines
#### Based on https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
####
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
    for i in range(len(xdata)):
        if x < xdata[i]:
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
    if type(xvals) == tuple:
        xmin, xmax = xvals
        xscale = ax.get_xscale()
        if xscale == "log":
            xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)

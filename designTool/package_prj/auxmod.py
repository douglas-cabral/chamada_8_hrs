'''
auxmod.py
this module contains auxiliary functions for Class 03

Ney Secco
2019-03-12
'''

# GENERAL IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS

def plot_contour(fun, ax,
                 xmin, xmax, ymin, ymax,
                 zmin, zmax, nlevels=15):

    '''
    This function generates a contour of a 2D function.

    INPUTS
    fun: function handle -> Handle to a 2D function f(x), where x
                            is a 2-element array.
    ax: axes handle -> Handle to the axes where we will plot the contour.
                       You can generate an axes handle using:
                       fig = plt.figure()
                       ax = plt.gca()
    xmin: real -> Lower bound for x
    xmax: real -> Upper bound for x
    ymin: real -> Lower bound for y
    ymax: real -> Upper bound for y
    zmin: real -> Lower bound for z
    zmax: real -> Upper bound for z
    nlevels: integer -> Number of contour lines. If nlevels=None, we only plot
                        f=0 line
    '''

    delta = 100
    x = np.linspace(xmin, xmax, delta)
    y = np.linspace(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    nx,ny = X.shape
    Z = np.zeros_like(X)
    for ii in range(nx):
        for jj in range(ny):
            Z[ii,jj] = fun([X[ii,jj], Y[ii,jj]])
    if nlevels is not None:
        ax.contour(X, Y, Z, np.logspace(np.log10(zmin), np.log10(zmax), nlevels))
    else:
        # Just plot f=0
        ax.contour(X, Y, Z, levels=[0])

    plt.axis('equal')

    ax.set_xlabel(r'$x_1$',fontsize=15)
    ax.set_ylabel(r'$x_2$',fontsize=15)

#==========================================

def plot_path(ax, xk, xopt=None):

    '''
    This function plots the optimizer path over the contour in ax

    INPUTS
    ax: axes handle -> Handle to the axes where we will plot the contour.
                       Use the same ax you used in plot_contour.
    xk: list of 2-elem arrays -> List of iterated design variables.
    xopt: 2-element array -> Coordinates of the analytical optimum.
    '''

    # Transform list into an array so that we can slice it vertically
    xk = np.array(xk)

    # Plot the optimizer path
    plt.plot(xk[:,0],xk[:,1],'--or')

    # Plot the starting point
    plt.plot(xk[0,0],xk[0,1],'og')

    # Plot the analytical optimum
    if xopt is not None:
        plt.plot(xopt[0],xopt[1],'ob')

#==========================================

def plot_history(ax, fk, xopt, fun):

    '''
    This function plots the objective function history.
    It also plots the distance to the analytical optimum.

    INPUTS
    ax: axes handle -> Handle to the axes where we will plot the contour.
                       Use the same ax you used in plot_contour.
    fk: list of real -> List of objective function values for the iterations.
    xopt: 2-element array -> Coordinates of the analytical optimum.
    fun: function handle -> Handle to a 2D function f(x), where x
                            is a 2-element array.
    '''

    # Transform list into an array so that we can make array operations
    fk = np.array(fk)

    # Compute the objective functoin value at the optimum
    fopt = fun(xopt)

    # Create a new window with 2 plots
    fig, axs = plt.subplots(2,1,sharex=True)

    # Plot the objective function history
    axs[0].plot(fk,'-o')

    axs[0].set_ylabel(r'$f$',fontsize=15)
    axs[0].get_xaxis().set_ticks([])
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)

    # Plot the distance to the optimum (residual)
    axs[1].semilogy(np.abs(fk-fopt),'-o')

    axs[1].set_xlabel('Iterations',fontsize=15)
    axs[1].set_ylabel(r'$|f-f^*|$',fontsize=15)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)

'''
OTIMIZAÇÃO MULTIDISCIPLINAR

Código com aplicações de ferramentas do Python para otimização restringida

Maj. Ney Sêcco 2025
'''

# IMPORTS
import numpy as np
from scipy.optimize import minimize
import auxmod as am
import matplotlib.pyplot as plt

# EXECUTION

# Define history list
Xlist = []
flist = []
g1list = []
g2list = []

# Define objective function

def objfun(X):

    # Split design variables
    x1 = X[0]
    x2 = X[1]

    # Compute objective function
    f = 1-np.exp(-0.1*(x1**2 + x2**2))

    # Save results for the plot
    Xlist.append(X)
    flist.append(f)

    return f

def objfungrad(X):

    # Split design variables
    x1 = X[0]
    x2 = X[1]

    # Compute objective function gradient
    gradf = np.array([np.exp(-0.1*(x1**2 + x2**2))*0.2*x1,
                      np.exp(-0.1*(x1**2 + x2**2))*0.2*x2])

    return gradf

# Define constraints

def confun(X):

    # Split design variables
    x1 = X[0]
    x2 = X[1]

    # Compute constraint functions
    g1 = x1 + x2 - 2
    g2 = x2 - 1

    # Save results for the plot
    g1list.append(g1)
    g2list.append(g2)

    return g1,g2

def confungrad(X):

    # Split design variables
    x1 = X[0]
    x2 = X[1]

    # Compute gradients of constraint functions
    gradg1 = np.array([1,
                       1])
    gradg2 = np.array([0,
                       1])

    return gradg1, gradg2


# Create list of constraints

con1 = {'type': 'ineq',
        'fun': confun,
        'jac': confungrad}

cons = [con1]

# Define starting point
X0 = np.array([0.5, 0.0])

# Define design variable bounds
bounds = [[-4.0, 4.0], [-4.0, 4.0]]

# Additional options
options = {'maxiter':200, 'ftol':1e-6}

# Run optimizer
result = minimize(objfun, X0, jac=objfungrad,
                  constraints=cons, bounds=bounds,
                  method='slsqp', options=options)

# Print results
print(result)
Xopt = result.x
xk = Xlist.copy()


# Plot optimization history
fig = plt.figure()
plt.subplot(311)
plt.plot(Xlist,'o-')
plt.ylabel('x',fontsize=20)
plt.subplot(312)
plt.plot(flist,'o-')
plt.ylabel('f',fontsize=20)
plt.subplot(313)
plt.plot(g1list,'o-')
plt.plot(g2list,'o-')
plt.plot([0,len(g1list)-1],[0,0],'gray',linewidth=0.5)
plt.ylabel('g',fontsize=20)
plt.xlabel('evaluations',fontsize=20)
plt.tight_layout()

# Plot results
fig = plt.figure()
ax = plt.gca()
am.plot_contour(objfun, ax,
                xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=10**-3, zmax=10**(-0.1))
am.plot_contour(lambda x: confun(x)[0], ax,
                xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=0, zmax=0, nlevels=None)
am.plot_contour(lambda x: confun(x)[1], ax,
                xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=0, zmax=0, nlevels=None)
am.plot_path(ax, xk, xopt=Xopt)

plt.show()
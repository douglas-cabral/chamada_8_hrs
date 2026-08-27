'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PROGRAMA DE ESPECIALIZAÇÃO EM ENGENHARIA AERONÁUTICA
OTIMIZAÇÃO MULTIDISCIPLINAR

Código com aplicações de ferramentas do Python para otimização
não restringida

Maj. Ney Sêcco 2025
'''

# IMPORTS
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EXECUTION

#==============================================

# Define number of input variables
nvar = 4

# Define objective function
def objfun(x):

    fun = np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

    return fun

def objfun_grad(x):

    grad = np.zeros(len(x))
    grad[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
    grad[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - 2*(1-x[1:-1]) + 200*(x[1:-1] - x[:-2]**2)
    grad[-1] = 200*(x[-1]-x[-2]**2)

    return grad

#==============================================

# Define initial guess
x0 = np.zeros(nvar)

### BFGS - With gradient

# Set optimization options
options = {'maxiter': 200000}

# Run the optimization algorithm
result = minimize(objfun, x0, jac = objfun_grad, method='BFGS', tol = 1e-6, options=options)

# Print results
print('\nBFGS with gradients')
print(result)


### BFGS - Without gradient

# Set optimization options
options = {'maxiter': 200000}

# Run the optimization algorithm
result = minimize(objfun, x0, method='BFGS', tol = 1e-6, options=options)

# Print results
print('\nBFGS without gradients')
print(result)


### NELDER-MEAD

# Set optimization options
options = {'maxiter': 200000, 'fatol': 1e-12, 'adaptive': True}

# Run the optimization algorithm
result = minimize(objfun, x0, method='Nelder-Mead', options=options)

# Print results
print('\nNelder-Mead')
print(result)


### DIFFERENTIAL EVOLUTION

# Define bounds to determine initial population
bounds = [[-5, 5]]*nvar

# Solve the optimization problem
result = differential_evolution(objfun, bounds, seed = 1, maxiter = 5000, polish = False, atol = 1e-12)

# Print results
print('\nDifferential Evolution')
print(result)


### GENETIC ALGORITHM

# Define bounds to determine initial population
bounds = [[-5, 5]]*nvar

# Solve the optimization problem
result = differential_evolution(objfun, bounds, seed = 1, maxiter = 5000, polish = False, atol = 1e-12)
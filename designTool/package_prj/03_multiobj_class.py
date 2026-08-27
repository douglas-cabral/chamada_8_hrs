'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PROGRAMA DE ESPECIALIZAÇÃO EM ENGENHARIA AERONÁUTICA
OTIMIZAÇÃO MULTIDISCIPLINAR

Código com aplicações de ferramentas do Python para otimização
multi-objetivo (pymoo 0.6.0)

Maj. Ney Sêcco 2025
'''

# IMPORTS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import matplotlib.pyplot as plt

# EXECUTION

# Define optimization problem
class MyProblem(ElementwiseProblem):

    def __init__(self):

        # Set general characteristics of the problem
        super().__init__(n_var=2, # Number of design variables
                         n_obj=2, # Number of objective functions
                         n_constr=2, # Number of constraints
                         xl=[-2, -2], # Lower bounds of design variables
                         xu=[2, 2]) # Upper bound of design variables

    def _evaluate(self, X, out, *args, **kwargs):

        # Split design variables
        x1 = X[0]
        x2 = X[1]

        # Compute objectives
        f1 = x1**2 + x2**2
        f2 = (x1-1)**2 + x2**2

        # Compute constraints
        g1 = 2*(x1-0.1) * (x1-0.9) / 0.18
        g2 = - 20*(x1-0.4) * (x1-0.6) / 4.8

        # Gather results
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

# Create an instance of the Problem
problem = MyProblem()

# Select optimization algorithm
algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

# Solve the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=True)

# Plot the Pareto in the objective space
fig = plt.figure()
plt.plot(res.F[:,0], res.F[:,1], 'o')
plt.title('Objective Space',fontsize=20)
plt.xlabel(r'$f_1$',fontsize=20)
plt.ylabel(r'$f_2$',fontsize=20)
plt.tight_layout()

# Plot the Pareto in the design space
fig = plt.figure()
plt.plot(res.X[:,0], res.X[:,1], 'o')
plt.title('Design Space',fontsize=20)
plt.xlabel(r'$x_1$',fontsize=20)
plt.ylabel(r'$x_2$',fontsize=20)
plt.axis('equal')
plt.tight_layout()

plt.show()
#fig.savefig('pareto_nsga.pdf')

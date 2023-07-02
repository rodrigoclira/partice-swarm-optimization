from PSO import PSO
from problem import *


if __name__ == '__main__':

    c1 = 2.05
    c2 = 2.05
    

    topology = "GLOBAL"

    dimensions = 5
    agents = 20
    problem = Sphere(dimensions)
    max_iter = 1000
    
    w_constant = lambda it: 0.7
    w_decay = lambda it: 1 - it * 0.8/max_iter

    pso = PSO(agents, dimensions, max_iter, problem, topology, c1, c2, w_decay)

    best_fit, best_pos, convergence = pso.optimize()

    print(best_fit)
    print(best_pos)
    print(convergence)

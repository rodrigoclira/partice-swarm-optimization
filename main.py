from PSO import PSO
from problem import *


if __name__ == '__main__':

    c1 = 2.05
    c2 = 2.05
    w = 0.7
    topology = "GLOBAL"

    dimensions = 5
    agents = 20
    problem = Sphere(dimensions)
    max_iter = 1000

    pso = PSO(agents, dimensions, max_iter, problem, topology, c1, c2, w)

    best_fit, best_pos, convergence = pso.optimize()

    print(best_fit)
    print(best_pos)
    print(convergence)

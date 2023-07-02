from particle import Particle
import numpy as np


class PSO:
    def __init__(self, num_agents, num_dimensions, max_iter, problem, topology, c1, c2, w):
        self.num_agents = num_agents
        self.num_dimensions = num_dimensions
        self.problem = problem
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.topology = topology
        self.swarm = []
        self.best_position = None
        self.best_fitness = np.inf
        self.best_particle_idx = -1
        self.curr_iter = 0
        self.convergence = []
        self._initialize_pop()

    def _initialize_pop(self):
        for idx in range(self.num_agents):
            velocity = np.random.uniform(low=-1,
                                         high=1,
                                         size=self.num_dimensions)
            position = np.random.uniform(low=self.problem.min_search_range[0],
                                         high=self.problem.max_search_range[0],
                                         size=self.num_dimensions)
            particle = Particle(idx=idx, init_pos=position, init_vel=velocity)
            fitness = self.problem.get_func_val(position)
            particle.fitness = fitness
            particle.particular_best_fitness = fitness
            self.swarm.append(particle)

            if fitness < self.best_fitness:
                self.best_position = position[:]
                self.best_particle_idx = idx
                self.best_fitness = fitness

    def _keeping_boundary_position(self):
        for particle in self.swarm:
            for dimension in range(self.num_dimensions):
                # Inverting positions out of search space
                if particle.position[dimension] < self.problem.min_search_range[0]:
                    particle.position[dimension] = self.problem.min_search_range[0]
                    particle.velocity[dimension] = - \
                        particle.velocity[dimension]

                if particle.position[dimension] > self.problem.max_search_range[0]:
                    particle.position[dimension] = self.problem.max_search_range[0]
                    particle.velocity[dimension] = - \
                        particle.velocity[dimension]

    def _keeping_boundary_velocity(self):
       # bound check for velocity
        # https://www.researchgate.net/post/How-do-I-set-maximum-and-minimum-velocity-value-in-a-PSO-algorithm
        maximum_velocity = (abs(
            self.problem.min_search_range[0]) + abs(self.problem.max_search_range[0])) * .5

        for particle in self.swarm:
            for dimension in range(self.num_dimensions):

                # Velocity higher or lower
                if particle.velocity[dimension] < -maximum_velocity:
                    random_factor = np.random.uniform(0.5, 1)
                    particle.velocity[dimension] = - \
                        maximum_velocity * random_factor

                elif particle.velocity[dimension] > maximum_velocity:
                    random_factor = np.random.uniform(0.5, 1)
                    particle.velocity[dimension] = maximum_velocity * \
                        random_factor

    def _get_best_neighbor_position(self, idx):

        if self.topology == "GLOBAL":
            return self.best_position
        else:
            raise NotImplementedError()

    def _update_velocity(self):
        for particle in self.swarm:
            r1 = np.random.random(size=self.num_dimensions)
            r2 = np.random.random(size=self.num_dimensions)

            best_neighbor_position = self._get_best_neighbor_position(
                particle.idx)

            cognitive = self.c1 * r1 * \
                (particle.particular_best_position - particle.position)

            social = self.c2 * r2 * \
                (best_neighbor_position - particle.position)

            new_velocity = self.w(self.curr_iter) * \
                particle.velocity + cognitive + social

            particle.velocity = new_velocity

    def _update_position(self):

        for particle in self.swarm:
            particle.position = particle.position[:] + \
                particle.velocity[:]  # PSO Equation

    def _evaluate_fitness(self):

        for particle in self.swarm:
            fitness = self.problem.get_func_val(particle.position)
            particle.fitness = fitness

            if fitness < self.best_fitness:
                self.best_position = particle.position[:]
                self.best_particle_idx = particle.idx
                self.best_fitness = fitness

            if fitness < particle.particular_best_fitness:
                particle.particular_best_position = particle.position[:]
                particle.particular_best_fitness = fitness

    def optimize(self):
        while self.curr_iter < self.max_iter:
            self._update_velocity()
            self._keeping_boundary_velocity()
            self._update_position()
            self._keeping_boundary_position()
            self._evaluate_fitness()
            self.convergence.append(self.best_fitness)
            self.curr_iter += 1

        return self.best_fitness, self.best_position, self.convergence

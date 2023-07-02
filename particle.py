import numpy as np


class Particle:
    def __init__(self, idx, init_pos, init_vel):
        self.idx = idx
        self.position = init_pos
        self.particular_best_position = init_pos
        self.particular_best_fitness = np.inf
        self.velocity = init_vel
        self.fitness = np.inf

    def __repr__(self) -> str:
        return f"{self.idx} | {self.position} | {self.fitness}"

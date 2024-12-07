import numpy as np
from car_rental.environment import CarRentalEnv


class ValueIterationAgent:
    def __init__(self, env, discount_factor=0.9):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        self.env = env
        self.discount_factor = discount_factor
        self.value_function = np.zeros((env.max_cars + 1, env.max_cars + 1))
        self.policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=int)

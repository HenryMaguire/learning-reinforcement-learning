import numpy as np
from blackjack.environment import BlackjackEnv


class BaseAgent:
    def __init__(self, env, discount_factor=0.9):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        self.env: BlackjackEnv = env
        self.discount_factor = discount_factor
        self.policy = np.zeros(env.state_space_dims, dtype=int)

    def train(self, max_iterations=1000, threshold=0.1):
        """
        Train the RL agent using a chosen algorithm.
        """
        raise NotImplementedError("Train method not implemented")

    def act(self, state):
        """
        Get the best action for a given state based on the current policy.
        :param state: Current state (tuple)
        :return: Action
        """
        return self.policy[state]

    def save_policy(self, filename):
        """
        Save the policy to a file.
        :param filename: Name of the file
        """
        np.save(filename, self.policy)
        print(f"Policy saved to {filename}")

    def load_policy(self, filename):
        """
        Load the policy from a file.
        :param filename: Name of the file
        """
        self.policy = np.load(filename)
        print(f"Policy loaded from {filename}")

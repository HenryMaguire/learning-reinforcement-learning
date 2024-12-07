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

    def evaluate_policy(self):
        """
        Evaluate the current policy by updating the value function.
        """
        print("Evaluating policy...")
        

    def improve_policy(self):
        """
        Improve the policy based on the current value function.
        """
        print("Improving policy...")
        # TODO: Implement policy improvement (e.g., greedy policy improvement).

    def train(self, max_iterations=1000):
        """
        Train the RL agent using a chosen algorithm.
        """
        print("Starting training...")
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}")
            self.evaluate_policy()
            self.improve_policy()

            # TODO: Add a stopping condition based on policy stability.

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


if __name__ == "__main__":
    # Initialize the environment
    env = CarRentalEnv()

    # Initialize the RL agent
    agent = PolicyIterationAgent(env)

    # Train the agent (policy iteration, value iteration, etc.)
    agent.train(max_iterations=1000)

    # Test the policy
    test_state = (10, 10)  # Example state
    best_action = agent.act(test_state)
    print(f"Best action for state {test_state}: {best_action}")

    # Save the policy
    agent.save_policy("car_rental_policy.npy")

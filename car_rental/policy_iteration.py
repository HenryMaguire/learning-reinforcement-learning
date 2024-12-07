import numpy as np
from car_rental.environment import CarRentalEnv
from tqdm import tqdm


class PolicyIterationAgent:
    def __init__(self, env, discount_factor=0.9):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        self.env: CarRentalEnv = env
        self.discount_factor = discount_factor
        self.value_function = np.zeros((env.max_cars + 1, env.max_cars + 1))
        self.policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=int)

    def evaluate_policy(self, max_iterations=100, threshold=0.1):
        """
        Evaluate the current policy by updating the value function.
        """
        print("Evaluating policy...")
        for iteration in range(max_iterations):
            delta = 0
            # Iterate over state space.
            for i in range(self.env.max_cars + 1):
                for j in range(self.env.max_cars + 1):
                    state = (i, j)
                    current_value = self.value_function[state]
                    action = self.policy[state]

                    new_value = 0
                    transitions = self.env.get_transition_probs(state, action)

                    for prob, next_state, reward in transitions:
                        next_value = self.value_function[next_state]
                        new_value += prob * (reward + self.discount_factor * next_value)

                    self.value_function[state] = new_value

                    delta = max(delta, abs(current_value - new_value))

            if delta < threshold:
                print(f"Delta converged: {delta}")
                break
            
            print(f"Iteration {iteration} complete")

    def improve_policy(self):
        """
        Improve the policy based on the current value function.
        """
        print("Improving policy...")

        policy_is_stable = True
        for i in range(self.env.max_cars + 1):
            for j in range(self.env.max_cars + 1):
                action_values = {}
                current_state = (i, j)
                old_action = self.policy[current_state]

                for action in self.env.get_valid_actions(current_state):
                    expected_value = 0
                    for prob, next_state, reward in env.get_transition_probs(
                        current_state, action
                    ):
                        expected_value += prob * (
                            reward
                            + self.discount_factor * self.value_function[next_state]
                        )

                    action_values[action] = expected_value

                best_action = max(action_values, key=action_values.get)

                self.policy[current_state] = best_action

                if old_action != best_action:
                    policy_is_stable = False

        return policy_is_stable

    def train(self, max_iterations=1000):
        """
        Train the RL agent using a chosen algorithm.
        """
        print("Starting training...")
        for _ in tqdm(range(max_iterations)):
            self.evaluate_policy()
            if self.improve_policy():
                print("Policy converged")
                break

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

import numpy as np
from car_rental.algorithms.base_agent import BaseAgent
from tqdm import tqdm


class PolicyIterationAgent(BaseAgent):
    def __init__(self, env, discount_factor=0.9):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        super().__init__(env, discount_factor)
        self.value_function = np.zeros(env.state_space_dims)

    def evaluate_policy(self, max_iterations=100, threshold=0.1):
        """
        Evaluate the current policy by updating the value function.
        """
        print("Evaluating policy...")
        for iteration in range(max_iterations):
            delta = 0
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
                    action_value = 0
                    for prob, next_state, reward in self.env.get_transition_probs(
                        current_state, action
                    ):
                        next_value = self.value_function[next_state]
                        action_value += prob * (
                            reward + self.discount_factor * next_value
                        )

                    action_values[action] = action_value

                best_action = max(action_values, key=action_values.get)

                self.policy[current_state] = best_action

                if old_action != best_action:
                    policy_is_stable = False

        return policy_is_stable

    def train(self, max_iterations=1000, threshold=0.1):
        """
        Train the RL agent using a chosen algorithm.
        """
        print("Starting training...")
        for _ in tqdm(range(max_iterations)):
            self.evaluate_policy(threshold=threshold)
            if self.improve_policy():
                print("Policy converged")
                break

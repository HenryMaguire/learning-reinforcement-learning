import numpy as np
from car_rental.environment import CarRentalEnv


class ValueIterationAgent:
    def __init__(self, env, discount_factor=0.9):
        self.env: CarRentalEnv = env
        self.discount_factor = discount_factor
        self.value_function = np.zeros((env.max_cars + 1, env.max_cars + 1))
        self.policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=int)

    def train_value_function(self, max_iterations=100, threshold=0.1):
        for iteration in range(max_iterations):
            delta = 0
            for i in range(self.env.max_cars + 1):
                for j in range(self.env.max_cars + 1):
                    state = (i, j)
                    old_value = self.value_function[state]

                    action_values = []
                    for action in self.env.get_valid_actions(state):
                        action_value = 0
                        for prob, new_state, reward in self.env.get_transition_probs(
                            state, action
                        ):
                            # In-place update of the value function (not synchronous).
                            next_value = self.value_function[new_state]
                            action_value += prob * (
                                reward + self.discount_factor * next_value
                            )
                        action_values.append(action_value)

                    best_value = max(action_values) if action_values else 0
                    self.value_function[state] = best_value

                    delta = max(delta, abs(old_value - best_value))

            if delta < threshold:
                print(f"Delta converged: {delta}")
                break

            print(f"Iteration {iteration} complete")

    def train_policy(self):
        for i in range(self.env.max_cars + 1):
            for j in range(self.env.max_cars + 1):
                state = (i, j)
                action_values = []
                for action in self.env.get_valid_actions(state):
                    action_value = 0
                    for prob, new_state, reward in self.env.get_transition_probs(
                        state, action
                    ):
                        next_value = self.value_function[new_state]
                        action_value += prob * (
                            reward + self.discount_factor * next_value
                        )
                    action_values.append((action_value, action))

                # Select the action with the highest value
                if action_values:
                    _, best_action = max(action_values)
                    self.policy[state] = best_action

    def train(self, max_iterations=100, threshold=0.1):
        self.train_value_function(max_iterations, threshold)
        self.train_policy()

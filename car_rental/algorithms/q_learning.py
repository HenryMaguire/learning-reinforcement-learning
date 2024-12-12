import numpy as np
from car_rental.environment import CarRentalEnv


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.01):
        self.env: CarRentalEnv = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.state_space_dims = (self.env.max_cars + 1, self.env.max_cars + 1)
        self.discount_factor = discount_factor
        self.q_values = {}  # A map from (state, action) -> value.
        self.policy = np.zeros(self.state_space_dims, dtype=int)

    def evaluate_policy(self, max_iterations=100, threshold=0.01):
        for iteration in range(max_iterations):
            delta = 0
            for i in range(self.state_space_dims[0]):
                for j in range(self.state_space_dims[1]):
                    state = (i, j)
                    action = self.policy[state]
                    current_q = self.q_values[(state, action)]
                    valid_actions = self.env.get_valid_actions(state)
                    q_values = [self.q_values.get((state, a), 0) for a in valid_actions]
                    best_q, best_action = max(zip(q_values, valid_actions))
                    next_action = (
                        best_action
                        if np.random.uniform() > self.epsilon
                        else np.random.choice(valid_actions)
                    )

                    next_state, reward = self.env.sample_step(state, next_action)
                    diff = reward + self.discount_factor * best_q - current_q
                    self.q_values[(state, action)] = (
                        current_q + self.learning_rate * diff
                    )

                    delta = max(delta, abs(diff))

            if delta < threshold:
                print(f"Converged after {iteration} iterations with delta: {delta}")
                break

    def optimise_policy(self):
        for i in range(self.state_space_dims[0]):
            for j in range(self.state_space_dims[1]):
                state = (i, j)
                valid_actions = self.env.get_valid_actions(state)
                q_values = [self.q_values.get((state, a), 0) for a in valid_actions]
                _, action = max(zip(q_values, valid_actions))
                self.policy[state] = action

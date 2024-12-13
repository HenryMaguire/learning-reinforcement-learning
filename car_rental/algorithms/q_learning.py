import numpy as np
from car_rental.algorithms.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, env, learning_rate=0.3, discount_factor=0.9, epsilon=0.1):
        super().__init__(env, discount_factor)
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.q_values = {}  # A map from (state, action) -> value.
        self.learning_rate = learning_rate

    def optimise_policy(self):
        for i in range(self.env.max_cars + 1):
            for j in range(self.env.max_cars + 1):
                state = (i, j)
                valid_actions = self.env.get_valid_actions(state)
                q_values = [self.q_values.get((state, a), 0) for a in valid_actions]
                _, action = max(zip(q_values, valid_actions))
                self.policy[state] = action

    def _train(self, max_episodes=5000, max_steps=200):
        """Direct Q-learning with improved exploration and learning"""
        # Decay epsilon over time
        initial_epsilon = self.epsilon
        epsilon_decay = 0.995
        min_epsilon = 0.01

        best_reward = float("-inf")
        for episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0

            # Decay epsilon
            self.epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay**episode))

            for step in range(max_steps):
                valid_actions = self.env.get_valid_actions(state)

                if np.random.uniform() < self.epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    q_values = [
                        self.q_values.get((state, a), 10.0) for a in valid_actions
                    ]  # Optimistic init
                    action = valid_actions[np.argmax(q_values)]

                next_state, reward = self.env.sample_step(state, action)
                if next_state is None:
                    break

                # Double Q-learning update to reduce overestimation
                next_valid_actions = self.env.get_valid_actions(next_state)
                next_q_values = [
                    self.q_values.get((next_state, a), 10.0) for a in next_valid_actions
                ]

                # Use separate max for action selection and value estimation
                best_action_idx = np.argmax(next_q_values)
                max_next_q = next_q_values[best_action_idx]

                current_q = self.q_values.get((state, action), 10.0)  # Optimistic init
                self.q_values[(state, action)] = current_q + self.learning_rate * (
                    reward + self.discount_factor * max_next_q - current_q
                )

                state = next_state
                total_reward += reward

            # Track and report progress
            if total_reward > best_reward:
                best_reward = total_reward
                self.optimise_policy()  # Update policy when we find better performance

            if episode % 100 == 0:  # Less frequent printing
                print(
                    f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}, Best: {best_reward:.2f}"
                )

    def train(self, max_iterations=1000, threshold=0.01):
        self._train(max_episodes=1000, max_steps=100)

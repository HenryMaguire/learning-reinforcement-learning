from collections import defaultdict

import numpy as np
from blackjack.algorithms.base_agent import BaseAgent
from blackjack.environment import BlackjackEnv


class TemporalDifferenceAgent(BaseAgent):
    def __init__(self, env, discount_factor=1.0, learning_rate=0.1, epsilon=0.1):
        super().__init__(env, discount_factor)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: np.zeros(len(self.env.action_space)))

    def train(self, num_episodes=1000, method="sarsa"):
        wins = 0
        episode_window = 10000
        win_rates = []

        for episode in range(num_episodes):
            self.epsilon = self.initial_epsilon / (1 + episode / 100000)
            self.learning_rate = self.initial_lr / (1 + episode / 100000)

            state = self.env.reset()
            done = False

            while not done:
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space[
                        np.random.randint(len(self.env.action_space))
                    ]
                else:
                    action = np.argmax(self.Q[state])

                next_state, reward, done = self.env.step(action)

                if method == "sarsa":
                    next_action = np.argmax(self.Q[next_state]) if not done else 0
                    target = (
                        reward + self.discount_factor * self.Q[next_state][next_action]
                    )
                else:
                    target = (
                        reward + self.discount_factor * np.max(self.Q[next_state])
                        if not done
                        else 0
                    )

                self.Q[state][action] += self.learning_rate * (
                    target - self.Q[state][action]
                )
                self.policy[state] = np.argmax(self.Q[state])
                state = next_state

            if reward == 1:  # Consider generalizing this check
                wins += 1

            if (episode + 1) % episode_window == 0:
                win_rate = wins / episode_window
                win_rates.append(win_rate)
                print(
                    f"Episode {episode + 1}/{num_episodes}, Win Rate: {win_rate:.3f}, "
                    f"Epsilon: {self.epsilon:.3f}, LR: {self.learning_rate:.3f}"
                )
                wins = 0

        return win_rates

    def train_sarsa(self, num_episodes=1000):
        return self.train(num_episodes, method="sarsa")

    def train_q_learning(self, num_episodes=1000):
        return self.train(num_episodes, method="q_learning")


if __name__ == "__main__":
    env = BlackjackEnv()
    agent = TemporalDifferenceAgent(env)
    agent.train_q_learning(num_episodes=1000000)
    agent.save_policy("temporal_difference_policy.npy")
    # agent.plot_value_function(
    #     show=True, save_path="temporal_difference_value_function.png"
    # )
    agent.plot_policy(show=True, save_path="temporal_difference_policy.png")

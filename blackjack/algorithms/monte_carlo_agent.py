from collections import defaultdict
import numpy as np
from blackjack.algorithms.base_agent import BaseAgent
from blackjack.environment import BlackjackEnv


class MonteCarloAgent(BaseAgent):
    def __init__(self, env, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Monte Carlo agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        :param epsilon: Exploration rate
        """
        super().__init__(env, discount_factor)
        self.Q = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        self.returns_count = defaultdict(lambda: np.zeros(2))
        self.epsilon = epsilon

    def train(self, num_episodes=10000):
        """
        Train the Monte Carlo agent using exploring starts
        """
        wins = 0
        episode_window = 10000
        win_rates = []

        for episode in range(num_episodes):
            episode_data = []
            state = self.env.reset()
            done = False

            while not done:
                # Update policy for current state
                self.policy[state] = np.argmax(self.Q[state])

                # Epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space[np.random.randint(2)]
                else:
                    action = self.policy.get(
                        state, self.env.action_space[np.random.randint(2)]
                    )

                next_state, reward, done = self.env.step(action)
                episode_data.append((state, action, reward))
                state = next_state

            # Track wins
            if reward == 1:
                wins += 1

            # Print win rate periodically
            if (episode + 1) % episode_window == 0:
                win_rate = wins / episode_window
                win_rates.append(win_rate)
                print(f"Episode {episode + 1}/{num_episodes}, Win Rate: {win_rate:.3f}")
                wins = 0

            # Update Q-values
            G = 0
            for state, action, reward in reversed(episode_data):
                G = reward + self.discount_factor * G
                self.returns_count[state][action] += 1
                self.Q[state][action] += (
                    G - self.Q[state][action]
                ) / self.returns_count[state][action]

        return win_rates


if __name__ == "__main__":
    env = BlackjackEnv()
    agent = MonteCarloAgent(env)
    agent.train(num_episodes=1000000)
    agent.save_policy("monte_carlo_policy.npy")
    agent.plot_value_function(show=True, save_path="monte_carlo_value_function.png")
    agent.plot_policy(show=True, save_path="monte_carlo_policy.png")

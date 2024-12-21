from collections import defaultdict
import numpy as np
from blackjack.environment import BlackjackEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MonteCarloAgent:
    def __init__(self, env, discount_factor=0.9, episilon=0.1):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        self.env: BlackjackEnv = env
        self.discount_factor = discount_factor
        # (s, a) -> Value
        self.Q = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        self.returns_count = defaultdict(lambda: np.zeros(2))
        self.policy = {}  # Current policy
        self.episilon = episilon

    def train(self, num_episodes=10000, threshold=0.1):
        """
        Train the MCRL agent
        """
        wins = 0
        episode_window = 10000  # Track win rate over last 1000 episodes
        win_rates = []

        for episode in range(num_episodes):
            episode_data = []
            state = self.env.reset()
            done = False

            while not done:
                self.policy[state] = np.argmax(self.Q[state])
                if np.random.rand() < self.episilon:
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

            if (episode + 1) % episode_window == 0:
                win_rate = wins / episode_window
                win_rates.append(win_rate)
                print(f"Episode {episode + 1}/{num_episodes}, Win Rate: {win_rate:.3f}")
                wins = 0  # Reset counter

            G = 0
            for state, action, reward in reversed(episode_data):
                G = reward + self.discount_factor * G
                self.returns_count[state][action] += 1
                self.Q[state][action] += (
                    G - self.Q[state][action]
                ) / self.returns_count[state][action]

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

    def plot_value_function(self, show=True, save_path=None):
        """
        Plot the value function as 3D surface plots, similar to Sutton & Barto.
        Shows player sum (12-21) vs dealer card (1-10) for both usable and non-usable ace.
        """
        player_sums = range(12, 22)
        dealer_cards = range(1, 11)

        # Create meshgrid for the plot
        X, Y = np.meshgrid(dealer_cards, player_sums)

        # Initialize value arrays for both ace scenarios
        Z_no_ace = np.zeros_like(X, dtype=float)
        Z_ace = np.zeros_like(X, dtype=float)

        # Calculate values
        for i, player in enumerate(player_sums):
            for j, dealer in enumerate(dealer_cards):
                # Get max value for each state instead of sum
                state_no_ace = (player, dealer, False)
                state_ace = (player, dealer, True)
                policy_action_no_ace = self.policy.get(state_no_ace, 0)
                policy_action_ace = self.policy.get(state_ace, 0)

                Z_no_ace[i][j] = self.Q[state_no_ace][policy_action_no_ace]
                Z_ace[i][j] = self.Q[state_ace][policy_action_ace]

        # Flip arrays vertically to match conventional representation
        Z_no_ace = np.flipud(Z_no_ace)
        Z_ace = np.flipud(Z_ace)

        # Create the plots
        fig = plt.figure(figsize=(15, 6))

        # Plot no usable ace
        ax1 = fig.add_subplot(121, projection="3d")
        surf1 = ax1.plot_surface(X, Y, Z_no_ace, cmap=plt.cm.viridis)
        ax1.set_title("Value Function\nNo Usable Ace")
        ax1.set_xlabel("Dealer Showing")
        ax1.set_ylabel("Player Sum")
        ax1.set_zlabel("Value")
        fig.colorbar(surf1)

        # Plot usable ace
        ax2 = fig.add_subplot(122, projection="3d")
        surf2 = ax2.plot_surface(X, Y, Z_ace, cmap=plt.cm.viridis)
        ax2.set_title("Value Function\nUsable Ace")
        ax2.set_xlabel("Dealer Showing")
        ax2.set_ylabel("Player Sum")
        ax2.set_zlabel("Value")
        fig.colorbar(surf2)

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_policy(self, show=True, save_path=None):
        """
        Plot the optimal policy as 2D grids.
        Shows player sum (12-21) vs dealer card (1-10) for both usable and non-usable ace.
        """
        player_sums = range(12, 22)
        dealer_cards = range(1, 11)

        # Initialize policy arrays for both ace scenarios
        policy_no_ace = np.zeros((len(player_sums), len(dealer_cards)))
        policy_ace = np.zeros((len(player_sums), len(dealer_cards)))

        # Get policy for each state
        for i, player in enumerate(player_sums):
            for j, dealer in enumerate(dealer_cards):
                state_no_ace = (player, dealer, False)
                state_ace = (player, dealer, True)
                # Get action with highest value (0 = stick, 1 = hit)
                policy_no_ace[i][j] = np.argmax(self.Q[state_no_ace])
                policy_ace[i][j] = np.argmax(self.Q[state_ace])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot no usable ace
        im1 = ax1.imshow(policy_no_ace, cmap="RdYlGn", aspect="auto")
        ax1.set_title("Optimal Policy\nNo Usable Ace")
        ax1.set_xlabel("Dealer Showing")
        ax1.set_ylabel("Player Sum")
        ax1.set_xticks(range(len(dealer_cards)))
        ax1.set_yticks(range(len(player_sums)))
        ax1.set_xticklabels(dealer_cards)
        ax1.set_yticklabels(player_sums)

        # Plot usable ace
        im2 = ax2.imshow(policy_ace, cmap="RdYlGn", aspect="auto")
        ax2.set_title("Optimal Policy\nUsable Ace")
        ax2.set_xlabel("Dealer Showing")
        ax2.set_ylabel("Player Sum")
        ax2.set_xticks(range(len(dealer_cards)))
        ax2.set_yticks(range(len(player_sums)))
        ax2.set_xticklabels(dealer_cards)
        ax2.set_yticklabels(player_sums)

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Stick", "Hit"])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    env = BlackjackEnv()
    agent = MonteCarloAgent(env)
    agent.train(num_episodes=1000000)
    agent.save_policy("monte_carlo_policy.npy")
    agent.plot_value_function(show=True, save_path="monte_carlo_value_function.png")
    agent.plot_policy(show=True, save_path="monte_carlo_policy.png")

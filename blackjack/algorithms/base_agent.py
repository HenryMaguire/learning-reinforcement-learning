import numpy as np
from blackjack.environment import BlackjackEnv
import matplotlib.pyplot as plt


class BaseAgent:
    def __init__(self, env, discount_factor=0.9):
        """
        Initialize the RL agent.
        :param env: The environment instance
        :param discount_factor: Discount factor (gamma)
        """
        self.env: BlackjackEnv = env
        self.discount_factor = discount_factor
        self.policy = {}

    def train(self, max_iterations=1000, threshold=0.1):
        """
        Train the RL agent using a chosen algorithm.
        """
        raise NotImplementedError("Train method not implemented")

    def save_policy(self, file_path: str):
        """
        Save the policy to a file.
        :param file_path: Path to the file where the policy will be saved
        """
        np.save(file_path, self.policy)
        print(f"Policy saved to {file_path}")

    def act(self, state: tuple[int, int, int]) -> int:
        """
        Select an action based on the current policy.
        :param state: The current state
        :return: The action to take
        """
        return self.policy.get(state, self.env.action_space[np.random.randint(2)])

    def plot_value_function(self, show=True, save_path=None):
        """
        Plot the value function as 3D surface plots.
        """
        player_sums = range(12, 22)
        dealer_cards = range(1, 11)
        X, Y = np.meshgrid(dealer_cards, player_sums)
        Z_no_ace = np.zeros_like(X, dtype=float)
        Z_ace = np.zeros_like(X, dtype=float)

        # Calculate values
        for i, player in enumerate(player_sums):
            for j, dealer in enumerate(dealer_cards):
                state_no_ace = (player, dealer, 0)
                state_ace = (player, dealer, 1)
                policy_action_no_ace = self.policy.get(state_no_ace, 0)
                policy_action_ace = self.policy.get(state_ace, 0)

                Z_no_ace[i][j] = self.Q[state_no_ace][policy_action_no_ace]
                Z_ace[i][j] = self.Q[state_ace][policy_action_ace]

        Z_no_ace = np.flipud(Z_no_ace)
        Z_ace = np.flipud(Z_ace)

        # Create plots
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
        plt.close()

    def plot_policy(self, show=True, save_path=None):
        """
        Plot the optimal policy as 2D grids.
        """
        player_sums = range(12, 22)
        dealer_cards = range(1, 11)

        policy_no_ace = np.zeros((len(player_sums), len(dealer_cards)))
        policy_ace = np.zeros((len(player_sums), len(dealer_cards)))

        for i, player in enumerate(player_sums):
            for j, dealer in enumerate(dealer_cards):
                state_no_ace = (player, dealer, 0)
                state_ace = (player, dealer, 1)
                policy_no_ace[i][j] = np.argmax(self.Q[state_no_ace])
                policy_ace[i][j] = np.argmax(self.Q[state_ace])

        # Flip arrays vertically
        policy_no_ace = np.flipud(policy_no_ace)
        policy_ace = np.flipud(policy_ace)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        im1 = ax1.imshow(policy_no_ace, cmap="RdYlGn", aspect="auto")
        ax1.set_title("Optimal Policy\nNo Usable Ace")
        ax1.set_xlabel("Dealer Showing")
        ax1.set_ylabel("Player Sum")
        ax1.set_xticks(range(len(dealer_cards)))
        ax1.set_yticks(range(len(player_sums)))
        ax1.set_xticklabels(dealer_cards)
        ax1.set_yticklabels(reversed(player_sums))

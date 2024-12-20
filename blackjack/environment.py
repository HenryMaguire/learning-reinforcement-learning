import numpy as np


class BlackjackEnv:
    def __init__(self):
        # State space: current sum (12-21), dealer's card (1-10), usable ace (0 or 1)
        self.state_space = (
            range(12, 22),  # Player's sum (12 to 21)
            range(1, 11),  # Dealer's showing card (1-10)
            [0, 1],  # Usable ace (0 or 1)
        )

        # Action space: stick (0) or twist (1)
        self.action_space = [0, 1]

        # Internal state
        self.current_sum = None
        self.dealer_card = None
        self.usable_ace = None

    def reset(self) -> tuple[int, int, int]:
        # Initialize player's sum to a random value between 12 and 21
        self.current_sum = np.random.randint(12, 22)

        # Dealer's showing card is between 1 and 10
        self.dealer_card = np.random.randint(1, 11)

        # Usable ace: randomly yes (1) or no (0)
        self.usable_ace = np.random.choice([0, 1])

        return self._get_obs()

    def _get_obs(self) -> tuple[int, int, int]:
        return (self.current_sum, self.dealer_card, self.usable_ace)

    def step(self, action) -> tuple[tuple, float, bool]:
        if action == 0:
            return self._stick()
        elif action == 1:
            return self._twist()

    def _stick(self) -> tuple[tuple, float, bool]:
        # Simulate dealer's behavior: dealer keeps hitting until 17 or above
        dealer_sum = self.dealer_card + np.random.randint(1, 11)
        while dealer_sum < 17:
            dealer_sum += np.random.randint(1, 11)

        if self.current_sum > 21:
            reward = -1
        elif dealer_sum > 21 or self.current_sum > dealer_sum:
            # Dealer busts or player wins
            reward = 1
        elif self.current_sum == dealer_sum:
            reward = 0
        else:
            reward = -1

        done = True
        return self._get_obs(), reward, done

    def _twist(self) -> tuple[tuple, float, bool]:
        new_card = np.random.randint(1, 11)
        print("New card:", new_card)
        self.current_sum += new_card

        if self.usable_ace and self.current_sum > 21:
            print("Using ace to avoid bust")
            self.current_sum -= 10
            self.usable_ace = 0

        if self.current_sum > 21:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return self._get_obs(), reward, done


if __name__ == "__main__":

    env = BlackjackEnv()
    state = env.reset()
    done = False
    print("Current sum:", state[0])
    print("Dealer's card:", state[1])
    print("Usable ace:", state[2])
    while not done:
        action = np.random.choice(env.action_space)  # Random action
        state, reward, done = env.step(action)
        print("Sum:", state[0])
        print(f"State: {state}, Reward: {reward}, Done: {done}")

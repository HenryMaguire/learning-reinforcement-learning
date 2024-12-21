import numpy as np


class BlackjackEnv:
    def __init__(self):
        # State space: current sum (12-21), dealer's card (1-10), usable ace (0 or 1)
        self.state_space = (
            range(12, 22),  # Player's sum (12 to 21)
            range(1, 11),  # Dealer's showing card (1-10, where 10 includes face cards)
            [0, 1],  # Usable ace (0 or 1)
        )
        self.state_space_dims = (10, 10, 2)

        # Action space: stick (0) or twist (1)
        self.action_space = [0, 1]

        # Internal state
        self.current_sum = None
        self.dealer_card = None
        self.usable_ace = None

    def reset(self) -> tuple[int, int, int]:
        # Deal two cards to player
        card1 = min(10, np.random.randint(1, 14))  # Ace=1, Face cards=10
        card2 = min(10, np.random.randint(1, 14))

        # Handle aces
        self.current_sum = card1 + card2
        self.usable_ace = 1 if (card1 == 1 or card2 == 1) else 0

        if self.usable_ace and self.current_sum + 10 <= 21:
            self.current_sum += 10

        if self.current_sum < 12:
            # Auto twist
            self._twist()

        # Dealer's up card
        self.dealer_card = self._draw_card()

        return self._get_obs()

    def _get_obs(self) -> tuple[int, int, int]:
        return (self.current_sum, self.dealer_card, self.usable_ace)

    def step(self, action) -> tuple[tuple, float, bool]:
        if action == 0:
            return self._stick()
        elif action == 1:
            return self._twist()

    def _stick(self) -> tuple[tuple, float, bool]:
        # Initial dealer hand
        dealer_sum = self.dealer_card
        dealer_ace = 1 if self.dealer_card == 1 else 0
        if dealer_ace:
            dealer_sum += 10

        # Keep hitting until 17 or above
        while dealer_sum < 17:
            card = self._draw_card()
            dealer_sum += card
            if card == 1 and dealer_ace == 1 and dealer_sum + 10 <= 21:
                dealer_sum += 10
                dealer_ace = 1
            elif dealer_ace and dealer_sum > 21:
                dealer_sum -= 10
                dealer_ace = 0

        if self.current_sum > 21:
            reward = -1
        elif dealer_sum > 21 or self.current_sum > dealer_sum:
            reward = 1
        elif self.current_sum == dealer_sum:
            reward = 0
        else:
            reward = -1

        done = True
        return self._get_obs(), reward, done

    def _twist(self) -> tuple[tuple, float, bool]:
        new_card = self._draw_card()
        self.current_sum += new_card

        if new_card == 1 and not self.usable_ace and self.current_sum + 10 <= 21:
            self.current_sum += 10
            self.usable_ace = 1
        elif self.usable_ace and self.current_sum > 21:
            self.current_sum -= 10
            self.usable_ace = 0

        if self.current_sum > 21:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return self._get_obs(), reward, done

    def _draw_card(self) -> int:
        card = np.random.randint(1, 14)  # 1-13 for Ace through King
        return min(10, card)  # Face cards = 10


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

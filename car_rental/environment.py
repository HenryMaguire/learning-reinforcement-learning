import math
import numpy as np


class CarRentalEnv:
    def __init__(self):
        self.max_cars = 20
        self.max_move = 5
        self.rental_reward = 10
        self.move_cost = 0
        self.rental_rates = [3, 4]  # Poisson parameters for rentals.
        self.return_rates = [3, 2]  # Poisson parameters for returns.

    def step(self, state, action):
        """
        Simulates one time step.
        :param state: (x, y) - cars at location 1 and 2
        :param action: Number of cars moved from location 1 to location 2 (can be negative)
        :return: new_state, reward
        """
        if action > self.max_move:
            raise ValueError(
                f"Action {action} is greater than the maximum move of {self.max_move}"
            )

        loc_1, loc_2 = state

        # Update the state
        loc_1 -= action
        loc_2 += action
        # Clamp the state to be between 0 and the maximum number of cars.
        loc_1 = max(0, min(self.max_cars, loc_1))
        loc_2 = max(0, min(self.max_cars, loc_2))
        movement_cost = abs(action) * self.move_cost

        # Sample the number of attempted car rentals and car returns.
        # Can never rent more cars out than are at the location.
        rentals_1 = min(loc_1, np.random.poisson(self.rental_rates[0]))
        rentals_2 = min(loc_2, np.random.poisson(self.rental_rates[1]))
        returns_1 = np.random.poisson(self.return_rates[0])
        returns_2 = np.random.poisson(self.return_rates[1])

        # Update the state again to account for rentals and returns.
        loc_1 = max(0, min(self.max_cars, loc_1 - rentals_1 + returns_1))
        loc_2 = max(0, min(self.max_cars, loc_2 - rentals_2 + returns_2))

        # Only get reward for new rentals.
        reward = (rentals_1 + rentals_2) * self.rental_reward - movement_cost

        return (loc_1, loc_2), reward

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        # Start with 10 cars at each location.
        return (10, 10)

    def get_transition_probs(self, state, action, min_prob=1e-4):
        """
        Get probabilities and rewards for all possible next states.
        Returns: list of (probability, next_state, reward) tuples
        """
        loc_1, loc_2 = state
        # First apply the action
        loc_1_after_move = max(0, min(self.max_cars, loc_1 - action))
        loc_2_after_move = max(0, min(self.max_cars, loc_2 + action))
        movement_cost = abs(action) * self.move_cost
        
        transitions = []
        # Consider reasonable ranges for Poisson distributions (e.g., mean ± 3 std dev)
        max_rental_1 = min(loc_1_after_move, int(self.rental_rates[0] + 3 * np.sqrt(self.rental_rates[0])))
        max_rental_2 = min(loc_2_after_move, int(self.rental_rates[1] + 3 * np.sqrt(self.rental_rates[1])))
        max_return_1 = int(self.return_rates[0] + 3 * np.sqrt(self.return_rates[0]))
        max_return_2 = int(self.return_rates[1] + 3 * np.sqrt(self.return_rates[1]))
        
        for r1 in range(max_rental_1 + 1):
            p_r1 = self._poisson_prob(self.rental_rates[0], r1)
            if p_r1 < min_prob:
                continue

            for r2 in range(max_rental_2 + 1):
                p_r2 = self._poisson_prob(self.rental_rates[1], r2)
                if p_r2 < min_prob:
                    continue

                for ret1 in range(max_return_1 + 1):
                    p_ret1 = self._poisson_prob(self.return_rates[0], ret1)
                    if p_ret1 < min_prob:
                        continue

                    for ret2 in range(max_return_2 + 1):
                        p_ret2 = self._poisson_prob(self.return_rates[1], ret2)
                        if p_ret2 < min_prob:
                            continue

                        # Calculate probability of this combination
                        p = p_r1 * p_r2 * p_ret1 * p_ret2

                        if p > min_prob:  # Ignore very unlikely transitions
                            # Calculate next state
                            next_loc1 = max(0, min(self.max_cars, 
                                                 loc_1_after_move - r1 + ret1))
                            next_loc2 = max(0, min(self.max_cars, 
                                                 loc_2_after_move - r2 + ret2))
                            # Calculate reward
                            reward = (r1 + r2) * self.rental_reward - movement_cost
                            transitions.append((p, (next_loc1, next_loc2), reward))
        
        return transitions

    @staticmethod
    def _poisson_prob(lambda_param, n):
        """Calculate Poisson probability mass function."""
        return (lambda_param ** n * np.exp(-lambda_param)) / math.factorial(n)

    def get_valid_actions(self, state):
        """Get list of valid actions for a state."""
        loc_1, loc_2 = state
        actions = []
        for a in range(-self.max_move, self.max_move + 1):
            if 0 <= loc_1 - a <= self.max_cars and 0 <= loc_2 + a <= self.max_cars:
                actions.append(a)
        return actions

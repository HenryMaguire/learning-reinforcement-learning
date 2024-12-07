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
            raise ValueError(f"Action {action} is greater than the maximum move of {self.max_move}")
        
        loc_1, loc_2 = state

        # Update the state
        loc_1 -= action
        loc_2 += action
        # Clamp the state to be between 0 and the maximum number of cars.
        loc_1 = max(0, min(self.max_cars, loc_1))
        loc_2 = max(0, min(self.max_cars, loc_2))
        movement_cost = abs(action) * self.move_cost

        # Sample the number of car rentals and car returns.
        rentals_1 = min(loc_1, np.random.poisson(self.rental_rates[0]))
        rentals_2 = min(loc_2, np.random.poisson(self.rental_rates[1]))
        returns_1 = np.random.poisson(self.return_rates[0])
        returns_2 = np.random.poisson(self.return_rates[1])

        # Update the state again to account for rentals and returns.
        loc_1 = max(0, min(self.max_cars, loc_1 - rentals_1 + returns_1))
        loc_2 = max(0, min(self.max_cars, loc_2 - rentals_2 + returns_2))
        
        # TODO: Calculate reward.
        
        reward = 0

        return (loc_1, loc_2), reward

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        # Start with 10 cars at each location.
        return (10, 10)

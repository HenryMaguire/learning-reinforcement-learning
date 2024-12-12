import numpy as np

from car_rental.helpers import TransitionCache, poisson_prob


class CarRentalEnv:
    def __init__(
        self,
        max_cars=20,
        max_move=5,
        rental_reward=10,
        move_cost=0,
        rental_rates=[3, 4],
        return_rates=[3, 2],
    ):
        self.max_cars = max_cars
        self.max_move = max_move
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        self.rental_rates = rental_rates  # Poisson parameters for rentals.
        self.return_rates = return_rates  # Poisson parameters for returns.
        self._transition_cache = TransitionCache()

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        # Start with 10 cars at each location.
        return (10, 10)

    def sample_step(self, state, action):
        probs, next_states, rewards = self.get_transition_probs(state, action=action)
        ids = range(len(next_states))
        choice = np.random.choice(ids, p=probs)
        return next_states[choice], rewards[choice]

    def get_transition_probs(self, state, action=0, min_prob=1e-4):
        """
        Get probabilities and rewards for all possible next states.
        Returns: list of (probability, next_state, reward) tuples
        """
        cached = self._transition_cache.get(state, action)
        if cached:
            return cached

        loc_1, loc_2 = state
        # First apply the action
        loc_1_after_move = max(0, min(self.max_cars, loc_1 - action))
        loc_2_after_move = max(0, min(self.max_cars, loc_2 + action))
        movement_cost = abs(action) * self.move_cost

        transitions = []
        # Consider reasonable ranges for Poisson distributions (e.g., mean ± 3 std dev)
        max_rental_1 = min(
            loc_1_after_move,
            int(self.rental_rates[0] + 3 * np.sqrt(self.rental_rates[0])),
        )
        max_rental_2 = min(
            loc_2_after_move,
            int(self.rental_rates[1] + 3 * np.sqrt(self.rental_rates[1])),
        )
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
                            next_loc1 = max(
                                0, min(self.max_cars, loc_1_after_move - r1 + ret1)
                            )
                            next_loc2 = max(
                                0, min(self.max_cars, loc_2_after_move - r2 + ret2)
                            )
                            # Calculate reward
                            reward = (r1 + r2) * self.rental_reward - movement_cost
                            transitions.append((p, (next_loc1, next_loc2), reward))

        self._transition_cache.set(state, action, transitions)
        return transitions

    def get_next_state(self, state, action):
        loc_1, loc_2 = state
        loc_1 -= action
        loc_2 += action
        return (loc_1, loc_2)

    @staticmethod
    def _poisson_prob(lambda_param, n):
        """Calculate Poisson probability mass function."""
        return poisson_prob(lambda_param, n)

    def get_valid_actions(self, state) -> list[int]:
        """Get list of valid actions for a state."""
        loc_1, loc_2 = state
        actions = []
        for a in range(-self.max_move, self.max_move + 1):
            if 0 <= loc_1 - a <= self.max_cars and 0 <= loc_2 + a <= self.max_cars:
                actions.append(a)
        return actions

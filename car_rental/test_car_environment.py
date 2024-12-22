import pytest
import numpy as np
from car_rental.environment import CarRentalEnv


@pytest.fixture
def env():
    return CarRentalEnv()


def test_initial_state(env):
    """Test that the initial state is set correctly."""
    state = env.reset()
    assert state == (10, 10)


def test_action_constraints(env):
    """Test that cars at each location stay within bounds after an action."""
    state = (20, 5)
    action = 10  # Attempt to move more cars than allowed
    with pytest.raises(ValueError):
        env.step(state, action)


def test_reward_calculation(env):
    """Test that the reward calculation is correct for a given state and action."""
    np.random.seed(42)  # Set seed for reproducibility
    state = (10, 10)
    action = 2  # Move 2 cars from location 1 to location 2
    _, reward = env.step(state, action)
    # Expected reward = rentals * rental_reward - action_cost
    # Rentals and returns are sampled from Poisson distribution
    assert reward >= 0  # Reward should not be negative


def test_rentals_and_returns(env):
    """Test that rentals and returns are properly handled, including randomness."""
    state = (5, 5)
    action = 0
    new_state, _ = env.step(state, action)
    # Ensure that cars remain within bounds [0, max_cars]
    assert 0 <= new_state[0] <= env.max_cars
    assert 0 <= new_state[1] <= env.max_cars


def test_no_negative_cars(env):
    """Test that the number of cars at a location never goes negative."""
    state = (1, 1)
    action = 5  # Move more cars than available
    new_state, _ = env.step(state, action)
    assert new_state[0] >= 0
    assert new_state[1] >= 0


def test_max_cars_constraint(env):
    """Test that the number of cars does not exceed the maximum limit."""
    state = (19, 19)
    action = 5
    new_state, _ = env.step(state, action)
    assert new_state[0] <= env.max_cars
    assert new_state[1] <= env.max_cars


def test_action_effect(env):
    """Test that moving cars correctly updates the state before rentals/returns."""
    state = (10, 10)
    action = -3  # Move 3 cars from location 2 to location 1

    # Mock the random sampling to test just the movement
    original_random = np.random.poisson
    # Temporarily make rentals and returns 3 (cancels out the action).
    np.random.poisson = lambda x: 3

    try:
        new_state, _ = env.step(state, action)
        # Only check that initial movement is correct (before rentals/returns)
        assert new_state == (13, 7)
    finally:
        # Restore the original random function
        np.random.poisson = original_random


def test_movement_cost(env):
    """Test that the movement cost is applied correctly to the reward."""
    state = (10, 10)
    action = 4  # Move 4 cars
    env.move_cost = 2
    _, reward = env.step(state, action)
    expected_cost = env.move_cost * abs(action)
    assert reward < env.rental_reward * (10 + 10) - expected_cost


def test_poisson_probability(env):
    """Test that Poisson probability calculations are correct."""
    # Test known values
    # P(X = 0) when lambda = 3 is approximately 0.0498
    assert abs(env._poisson_prob(3, 0) - 0.0498) < 0.001
    # P(X = 3) when lambda = 3 is approximately 0.224
    assert abs(env._poisson_prob(3, 3) - 0.224) < 0.001
    # Probability should never be negative
    assert env._poisson_prob(4, 2) >= 0


def test_transition_probabilities(env):
    """Test that transition probabilities sum to approximately 1."""
    state = (10, 10)
    action = 2
    
    transitions = env.get_transition_probs(state, action)
    
    # Sum of all probabilities should be approximately 1
    total_prob = sum(prob for prob, _, _ in transitions)
    assert abs(1 - total_prob) < 0.02
    
    # Check that all next states are valid
    for _, next_state, _ in transitions:
        assert 0 <= next_state[0] <= env.max_cars
        assert 0 <= next_state[1] <= env.max_cars
        
    # Check that all probabilities are positive
    for prob, _, _ in transitions:
        assert prob > 0


def test_valid_actions(env):
    """Test that valid actions are correctly identified."""
    # Test edge cases
    state = (0, 0)
    actions = env.get_valid_actions(state)
    assert 0 in actions  # No movement should always be valid
    assert -env.max_move not in actions  # Can't move cars from empty location
    
    state = (env.max_cars, env.max_cars)
    actions = env.get_valid_actions(state)
    assert 0 in actions
    assert env.max_move not in actions  # Can't exceed max cars
    
    # Test middle state
    state = (10, 10)
    actions = env.get_valid_actions(state)
    assert -env.max_move in actions
    assert 0 in actions
    assert env.max_move in actions
    assert len(actions) == 2 * env.max_move + 1  # All actions should be valid


def test_transition_rewards(env):
    """Test that rewards are calculated correctly in transitions."""
    state = (5, 5)
    action = 2
    movement_cost = abs(action) * env.move_cost
    
    transitions = env.get_transition_probs(state, action)
    
    for _, _, reward in transitions:
        # Reward should be a multiple of rental reward minus movement cost
        assert (reward + movement_cost) % env.rental_reward == 0
        # Maximum possible reward is renting all cars
        assert reward <= (state[0] + state[1]) * env.rental_reward



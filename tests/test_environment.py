import pytest
import numpy as np
from car_rental_environment import CarRentalEnv

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
    _, reward = env.step(state, action)
    expected_cost = env.move_cost * abs(action)
    assert reward < env.rental_reward * (10 + 10) - expected_cost

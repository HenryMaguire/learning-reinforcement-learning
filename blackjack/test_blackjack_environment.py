import pytest
import numpy as np
from blackjack.environment import BlackjackEnv


@pytest.fixture
def env():
    return BlackjackEnv()


def test_init(env):
    assert env.action_space == [0, 1]
    assert env.state_space[0] == range(12, 22)  # player sum
    assert env.state_space[1] == range(1, 11)  # dealer card
    assert env.state_space[2] == [0, 1]  # usable ace


def test_reset(env):
    state = env.reset()
    sum, dealer, ace = state

    assert 12 <= sum <= 21
    assert 1 <= dealer <= 10
    assert ace in [0, 1]


def test_stick_win(env, monkeypatch):
    # Mock dealer getting a bad hand (22)
    def mock_randint(a, b):
        return 12  # Dealer will get 12 + dealer_card, causing bust

    monkeypatch.setattr(np.random, "randint", mock_randint)

    env.current_sum = 20
    env.dealer_card = 10
    env.usable_ace = 0

    state, reward, done = env._stick()
    assert reward == 1  # Player should win
    assert done == True


def test_stick_lose(env, monkeypatch):
    # Mock dealer getting 20
    def mock_randint(a, b):
        return 10

    monkeypatch.setattr(np.random, "randint", mock_randint)

    env.current_sum = 18
    env.dealer_card = 10
    env.usable_ace = 0

    state, reward, done = env._stick()
    assert reward == -1
    assert done == True

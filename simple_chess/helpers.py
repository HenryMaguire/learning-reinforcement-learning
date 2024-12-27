from typing import Callable
import gymnasium as gym
import numpy as np
import torch
from simple_chess.environment import ChessEnv


def make_env(
    max_game_length: int,
    white_score_weight: float,
    black_score_weight: float,
) -> Callable:
    """Create a callable that creates an environment with the given parameters."""

    def _init() -> gym.Env:
        env = ChessEnv(
            max_game_length=max_game_length,
            white_score_weight=white_score_weight,
            black_score_weight=black_score_weight,
        )
        return env

    return _init


from gymnasium.vector import AsyncVectorEnv


class ChessVectorEnv(AsyncVectorEnv):
    def get_random_action(self, as_index: bool = True):
        return self.call("get_random_action", as_index=as_index)

    def get_legal_moves_mask(self):
        return np.array(self.call("_get_legal_moves_mask"))

    def get_win_lose_draw(self):
        return np.array(self.call("get_win_lose_draw"))


def to_tensor(data, device):
    if isinstance(data, list):
        if isinstance(data[0], np.ndarray):
            data = np.array(data)
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
    return torch.tensor(data, dtype=torch.float32).to(device)


def check_significance_of_improvement(model_wins: int, total_games: int) -> None:
    """
    Quickly tests if your model's win rate is above 50% using a one-proportion z-test.
    """
    import math
    from statsmodels.stats.proportion import proportions_ztest

    # Null hypothesis is that the true win rate p = 0.5
    stat, p_value = proportions_ztest(model_wins, total_games, value=0.5)
    win_rate = model_wins / total_games

    print(f"Model win rate: {win_rate*100:.2f}%")
    print(f"Test statistic: {stat:.3f}, p-value: {p_value:.6f}")
    if p_value < 0.05:
        if win_rate > 0.5:
            print("Statistically significant improvement over random at p<0.05")
        else:
            print("Statistically significant sub-random performance at p<0.05")
    else:
        print("Not enough evidence to conclude it's better than random.")

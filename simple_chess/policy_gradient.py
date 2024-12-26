from copy import deepcopy
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from simple_chess.base_policy_model import PolicyNetwork
from simple_chess.environment import ChessEnv
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# hyperparameters
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
LOAD_SAVED_MODEL = False


def make_env(
    max_game_length: int, white_score_weight: float, black_score_weight: float
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


def calculate_discounted_rewards(reward_history):
    """Compute discounted rewards with baseline normalization."""
    discounted_rewards = np.zeros_like(reward_history)
    cumulative_reward = 0

    for t in reversed(range(len(reward_history))):
        cumulative_reward = cumulative_reward * DISCOUNT_FACTOR + reward_history[t]
        discounted_rewards[t] = cumulative_reward

    # Normalize rewards to help with training stability
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards) + 1e-8
    return discounted_rewards


def _tensor(data, device):
    if isinstance(data, list):
        if isinstance(data[0], np.ndarray):
            data = np.array(data)
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
    return torch.tensor(data, dtype=torch.float32).to(device)


class LossWeights(nn.Module):
    def __init__(
        self,
        init_entropy_weight=0.01,
        init_invalid_weight=0.5,
        init_gradient_weight=0.01,
    ):
        super().__init__()
        self.entropy_weight = nn.Parameter(torch.tensor(init_entropy_weight))
        self.invalid_weight = nn.Parameter(torch.tensor(init_invalid_weight))
        self._raw_gradient_weight = nn.Parameter(
            torch.tensor(np.log(init_gradient_weight), dtype=torch.float32)
        )

    @property
    def gradient_weight(self):
        return torch.exp(self._raw_gradient_weight)


def reinforce(
    policy,
    episodes,
    alpha=5e-4,
    gamma=0.99,
    num_envs=10,
    batch_size=32,
    max_gradient_norm=2.0,
):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    loss_weights = LossWeights().to(device)
    policy = policy.to(device)
    optim = AdamW(list(policy.parameters()) + list(loss_weights.parameters()), lr=alpha)
    scheduler = StepLR(optim, step_size=100, gamma=0.5)

    stats = {"PG Loss": [], "Returns": [], "Game Lengths": [], "WinLoss": []}
    score_weight = 0.5
    game_length = 50
    # max_game_length = 70
    max_reward = 0
    for episode_batch in tqdm(range(1, episodes + 1)):
        # if episode_batch % 20 == 0:
        #     game_length = min(game_length + 10, max_game_length)

        env_fns = [
            make_env(game_length, score_weight, score_weight) for _ in range(num_envs)
        ]
        envs = AsyncVectorEnv(env_fns)
        states, _ = envs.reset()  # Reset the environment
        dones = [False] * num_envs
        batch_transitions = [[] for _ in range(num_envs)]
        while not all(dones):
            # Shape (4, 12, 8, 8)
            position_tensor = _tensor(states, device)
            action_probs = policy(position_tensor)
            illegal_mask = _tensor(
                np.array(envs.get_attr("_get_legal_moves_mask")),
                device,
            )
            masked_action_scores = action_probs * illegal_mask
            actions = []
            for i in range(num_envs):
                if masked_action_scores[i].sum() == 0:
                    # Take the illegal move and get punished.
                    action = action_probs[i].multinomial(1).detach()
                else:
                    masked_action_probs = (
                        masked_action_scores[i] / masked_action_scores[i].sum()
                    )
                    action = masked_action_probs.multinomial(1).detach()
                actions.append(action.item())

            next_states, rewards, new_dones, _, _ = envs.step(actions)
            for i in range(num_envs):
                batch_transitions[i].append(
                    (states[i], actions[i], rewards[i], illegal_mask[i])
                )
                if new_dones[i]:
                    dones[i] = True

            states = next_states

        all_states = []
        all_actions = []
        all_returns = []
        all_legal_masks = []
        total_reward = 0
        win_loss = 0
        for env_idx, transitions in enumerate(batch_transitions):
            _rewards = [t[2] for t in transitions]
            mean_reward = np.mean(_rewards)
            std_reward = np.std(_rewards)
            total_reward += sum(_rewards)
            win_loss += sum(_rewards) > 0 - sum(_rewards) <= 0

            G = 0
            # Moving from the last timestep, calculate the discounted return.
            for t, (state_t, action_t, reward_t, legal_mask_t) in reversed(
                list(enumerate(transitions))
            ):
                reward_t = (reward_t - mean_reward) / (std_reward + 1e-6)
                G = reward_t + gamma * G
                all_states.append(state_t)
                all_actions.append(action_t)
                all_returns.append(G)
                all_legal_masks.append(legal_mask_t)

        # Prepare tensors for training
        states = _tensor(all_states, device)
        actions = _tensor(all_actions, device)
        returns = _tensor(all_returns, device)
        legal_masks = _tensor(all_legal_masks, device)

        # Shuffle the data
        indices = torch.randperm(len(states))
        states = states[indices]
        actions = actions[indices]
        returns = returns[indices]
        legal_masks = legal_masks[indices]
        print(
            "Rewards | Loss | PG Loss | Invalid | Entropy | gNorm | bEntropy | bInvalid | bGradient"
        )
        for i in range(0, len(states), batch_size):
            states_batch = states[i : i + batch_size]
            # Actions taken by the current policy.
            actions_batch = actions[i : i + batch_size]
            returns_batch = returns[i : i + batch_size]
            legal_masks_batch = legal_masks[i : i + batch_size]
            # Probability distribution over all actions.
            probs = policy(states_batch)
            log_probs = torch.log(probs + 1e-6)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
            action_log_prob = log_probs.gather(1, actions_batch.unsqueeze(1).long())
            pg_loss = -(returns_batch * action_log_prob).mean()
            entropy_loss = -entropy.mean()

            invalid_probs = probs * (1 - legal_masks_batch)
            invalid_move_loss = torch.log(invalid_probs.sum(dim=1)).mean()
            total_loss = (
                pg_loss
                + loss_weights.invalid_weight * invalid_move_loss
                + 0.02 * entropy_loss
            )
            total_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=max_gradient_norm
            )
            print(
                f"{returns_batch.mean().item():.5f} | {total_loss.item():.5f} | {pg_loss.item():.5f} | {invalid_move_loss.item():.5f} | {entropy_loss.item():.5f} | {total_norm:.3f} | {loss_weights.entropy_weight.item():.3f} | {loss_weights.invalid_weight.item():.3f}| {loss_weights.gradient_weight.item():.3f}"
            )
            stats["PG Loss"].append(total_loss.item())
            stats["Returns"].append(returns_batch.mean().item())
        optim.step()
        optim.zero_grad()
        scheduler.step()
        max_reward = max(max_reward, total_reward / num_envs)

        print(
            f"Average Reward: {total_reward / num_envs:.2f} Max Reward: {max_reward:.2f} Learning Rate: {optim.param_groups[0]['lr']:.4f} WinLoss: {win_loss}"
        )
    envs.close()
    return stats


if __name__ == "__main__":
    policy_network = PolicyNetwork()
    stats = reinforce(policy_network, 200, num_envs=10)
    # print(stats)

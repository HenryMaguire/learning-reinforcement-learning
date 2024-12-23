from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from tqdm import tqdm
from simple_chess.environment import ChessEnv
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# hyperparameters
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
LOAD_SAVED_MODEL = False


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Example CNN structure - you'll want to replace this
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3)

        # 4096 = 64*64 possible moves
        self.move_predictor = nn.Linear(256 * 6 * 6, 4096)

    def forward(self, board_position):
        features = torch.relu(self.conv1(board_position))
        features = torch.relu(self.conv2(features))
        features = torch.relu(self.conv3(features))
        features_flat = features.view(-1, 256 * 6 * 6)
        move_probabilities = torch.softmax(self.move_predictor(features_flat), dim=1)
        return move_probabilities


def make_env(max_game_length: int, score_weight: float) -> Callable:
    """Create a callable that creates an environment with the given parameters."""

    def _init() -> gym.Env:
        env = ChessEnv(max_game_length=max_game_length, score_weight=score_weight)
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
    return torch.tensor(data, dtype=torch.float32).to(device)


def reinforce(
    policy,
    episodes,
    alpha=1e-3,
    gamma=0.99,
    beta=0.01,
    num_envs=10,
    batch_size=64,
    max_gradient_norm=10.0,
):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    policy = policy.to(device)
    optim = AdamW(policy.parameters(), lr=alpha)
    scheduler = StepLR(optim, step_size=10, gamma=0.7)

    stats = {"PG Loss": [], "Returns": [], "Game Lengths": []}
    score_weight = 1
    game_length = 20
    max_reward = 0
    for episode_batch in tqdm(range(1, episodes + 1)):
        if episode_batch % 10 == 0:
            # score_weight *= 0.9
            game_length = min(game_length + 5, 50)

        env_fns = [
            make_env(min(game_length, 50), score_weight) for _ in range(num_envs)
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
                batch_transitions[i].append((states[i], actions[i], rewards[i]))
                if new_dones[i]:
                    dones[i] = True

            states = next_states

        all_states = []
        all_actions = []
        all_returns = []

        total_reward = 0
        for env_idx, transitions in enumerate(batch_transitions):
            _rewards = [t[2] for t in transitions]
            mean_reward = np.mean(_rewards)
            std_reward = np.std(_rewards)
            total_reward += sum(_rewards)

            G = 0
            for t, (state_t, action_t, reward_t) in reversed(
                list(enumerate(transitions))
            ):
                reward_t = (reward_t - mean_reward) / (std_reward + 1e-6)
                G = reward_t + gamma * G  # Calculate discounted return
                all_states.append(state_t)
                all_actions.append(action_t)
                all_returns.append(G)

        # Prepare tensors for training
        states = _tensor(all_states, device)
        actions = _tensor(all_actions, device)
        returns = _tensor(all_returns, device)
        for i in range(0, len(states), batch_size):
            states_batch = states[i : i + batch_size]
            actions_batch = actions[i : i + batch_size]
            returns_batch = returns[i : i + batch_size]
            probs = policy(states_batch)
            log_probs = torch.log(probs + 1e-6)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
            action_log_prob = log_probs.gather(1, actions_batch.unsqueeze(1).long())
            pg_loss = -(returns_batch * action_log_prob).mean()
            entropy_loss = entropy.mean()
            total_loss = pg_loss - beta * entropy_loss

            optim.zero_grad()
            total_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=max_gradient_norm
            )
            optim.step()
        scheduler.step()
        max_reward = max(max_reward, total_reward / num_envs)

        print(
            f"Average Reward: {total_reward / num_envs:.2f} Max Reward: {max_reward:.2f} Learning Rate: {optim.param_groups[0]['lr']:.4f}"
        )
    envs.close()
    return stats


if __name__ == "__main__":
    policy_network = PolicyNetwork()
    stats = reinforce(policy_network, 200, num_envs=10)
    print(stats)

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from simple_chess.base_policy_model import ChessCNN, ChessMLP
from simple_chess.helpers import check_significance_of_improvement, make_env, to_tensor
from torch.optim.lr_scheduler import StepLR
from gymnasium.vector import AsyncVectorEnv


class LossWeights(nn.Module):
    def __init__(
        self,
        init_entropy_weight=0.01,
        init_invalid_weight=0.1,
    ):
        super().__init__()
        self.entropy_weight = nn.Parameter(torch.tensor(init_entropy_weight))
        self.invalid_weight = nn.Parameter(torch.tensor(init_invalid_weight))


def reinforce(
    policy,
    episodes,
    alpha=1e-4,
    gamma=0.99,
    entropy_weight=0.02,
    invalid_weight=1.0,
    score_weight=0.5,
    num_envs=10,
    batch_size=128,
    epsilon=0.05,
    max_gradient_norm=1.0,
):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    loss_weights = LossWeights(
        init_entropy_weight=entropy_weight,
        init_invalid_weight=invalid_weight,
    ).to(device)

    policy = policy.to(device)
    optim = AdamW(list(policy.parameters()) + list(loss_weights.parameters()), lr=alpha)
    scheduler = StepLR(optim, step_size=100, gamma=0.8)

    stats = {"PG Loss": [], "Returns": [], "Game Lengths": [], "WinLoss": []}
    game_length = 50
    max_reward = 0
    model_wins = 0
    total_games = 0
    for episode_batch in tqdm(range(1, episodes + 1)):
        env_fns = [
            make_env(game_length, score_weight, score_weight) for _ in range(num_envs)
        ]
        envs = AsyncVectorEnv(env_fns)
        states, _ = envs.reset()  # Reset the environment
        dones = [False] * num_envs
        batch_transitions = [[] for _ in range(num_envs)]
        while not all(dones):
            # Shape (4, 12, 8, 8)
            position_tensor = to_tensor(states, device)
            with torch.no_grad():
                action_probs = policy(position_tensor)
            illegal_mask = to_tensor(
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
                    if torch.rand(1) < epsilon:
                        action = masked_action_probs.multinomial(1).detach()
                    else:
                        action = masked_action_probs.argmax().detach()

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
            # Net positive reward is considered a win.
            win_loss += int((sum(_rewards) + 5) > 0)
            total_games += 1
            model_wins += int((sum(_rewards) + 5) > 0)

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
        states = to_tensor(all_states, device)
        actions = to_tensor(all_actions, device)
        returns = to_tensor(all_returns, device)
        legal_masks = to_tensor(all_legal_masks, device)

        # Shuffle the data
        indices = torch.randperm(len(states))
        states = states[indices]
        actions = actions[indices]
        returns = returns[indices]
        legal_masks = legal_masks[indices]
        print(
            "Rewards | Loss | PG Loss | Invalid | Entropy | gNorm | bEntropy | bInvalid | bGradient"
        )
        num_batches = len(states) // batch_size
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
                # + loss_weights.invalid_weight * invalid_move_loss
                + entropy_weight * entropy_loss
            )
            # total_loss /= num_batches
            total_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=max_gradient_norm
            )
            print(
                f"{returns_batch.mean().item():.5f} | {total_loss.item():.5f} | {pg_loss.item():.5f} | {invalid_move_loss.item():.5f} | {entropy_loss.item():.5f} | {total_norm:.3f}"
            )
            stats["PG Loss"].append(total_loss.item())
            stats["Returns"].append(returns_batch.mean().item())

            optim.step()
            optim.zero_grad()

        scheduler.step()
        max_reward = max(max_reward, total_reward / num_envs)
        print(num_batches)
        print(
            f"Average Ep. Reward: {total_reward / num_envs:.2f} Max Ep. Reward: {max_reward:.2f} Learning Rate: {optim.param_groups[0]['lr']:.4f} Ep. WinLoss: {win_loss / num_envs:.2f} Norm: {total_norm:.3f}"
        )
        if episode_batch % 20 == 0:
            check_significance_of_improvement(model_wins, total_games)
            model_wins = 0
            total_games = 0

    envs.close()
    return stats


def plot_stats(stats):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot PG Loss
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("PG Loss", color="tab:blue")
    ax1.plot(stats["PG Loss"], color="tab:blue", alpha=0.6, label="PG Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot Returns on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Returns", color="tab:orange")
    ax2.plot(stats["Returns"], color="tab:orange", alpha=0.6, label="Returns")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Training Progress")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    policy_network = ChessCNN()
    # Load a model that slightly knows what moves are legal (~40% legal move prob).
    # policy_network.load_state_dict(torch.load("pretrained_policy.pt"))
    num_cpus = os.cpu_count()
    print(f"Using {num_cpus} CPUs")
    stats = reinforce(policy_network, 2000, num_envs=num_cpus)
    plot_stats(stats)

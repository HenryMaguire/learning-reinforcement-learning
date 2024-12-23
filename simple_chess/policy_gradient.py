import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from tqdm import tqdm
from simple_chess.environment import ChessEnvironment


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

        self.move_predictor = nn.Linear(
            256 * 6 * 6, 4096
        )  # 4096 = 64*64 possible moves

    def forward(self, board_position):
        features = torch.relu(self.conv1(board_position))
        features = torch.relu(self.conv2(features))
        features = torch.relu(self.conv3(features))
        features_flat = features.view(-1, 256 * 6 * 6)
        move_probabilities = torch.softmax(self.move_predictor(features_flat), dim=1)
        return move_probabilities


# Initialize network and optimizer
policy_network = PolicyNetwork()
optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)

if LOAD_SAVED_MODEL:
    saved_checkpoint = torch.load("chess_model.pt")
    policy_network.load_state_dict(saved_checkpoint["model_state"])
    optimizer.load_state_dict(saved_checkpoint["optimizer_state"])


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


def reinforce(policy, episodes, alpha=1e-4, gamma=0.99, beta=0.01):
    optim = AdamW(policy.parameters(), lr=alpha)
    stats = {"PG Loss": [], "Returns": []}
    result_weight = 0
    game_length = 10
    for episode in tqdm(range(1, episodes + 1)):
        if episode % 10 == 0:
            result_weight += 0.1
            game_length += 10
        chess_env = ChessEnvironment(
            max_game_length=min(game_length, 50),
            result_weight=min(result_weight, 1),
        )
        state = chess_env.reset()  # Reset the environment
        done = False
        transitions = []

        while not done:
            # Convert state to tensor and get action probabilities
            position_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(
                0
            )  # Shape (1, 12, 8, 8)
            action_probs = policy(position_tensor)
            illegal_mask = torch.tensor(
                chess_env._get_legal_moves_mask(), dtype=torch.float32
            )
            masked_action_scores = action_probs * illegal_mask
            masked_action_probs = masked_action_scores / masked_action_scores.sum()
            if masked_action_scores.sum() == 0:
                # Take the illegal move and get the reward.
                action = action_probs.multinomial(1).detach()
            else:
                action = masked_action_probs.multinomial(1).detach()

            next_state, reward, done, _ = chess_env.step(action.item())
            transitions.append((state, action, reward))
            state = next_state

        # Calculate returns and policy gradient loss
        rewards = [reward for _, _, reward in transitions]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        G = 0
        cumulative_loss = 0
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            reward_t = (reward_t - mean_reward) / (std_reward + 1e-6)
            G = reward_t + gamma * G  # Calculate discounted return
            position_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0)
            probs_t = policy(position_tensor)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)

            # Calculate entropy for regularization
            entropy_t = -torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
            gamma_t = gamma**t
            pg_loss_t = -gamma_t * action_log_prob_t * G  # Policy gradient loss
            total_loss_t = (pg_loss_t - beta * entropy_t).mean()  # Total loss
            cumulative_loss += total_loss_t
            # Update policy
            optim.zero_grad()
            total_loss_t.backward()
            optim.step()
        print(
            f"Episode {episode}: PG Loss: {cumulative_loss.item():.2f}, Return: {sum(rewards):.2f} Game Length: {len(transitions)}"
        )
        # Track statistics
        stats["PG Loss"].append(cumulative_loss.item())
        stats["Returns"].append(sum(rewards))

    return stats


stats = reinforce(policy_network, 200)
print(stats)

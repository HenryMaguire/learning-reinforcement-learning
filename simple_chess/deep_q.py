from collections import deque
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from simple_chess.base_policy_model import ChessCNN
from simple_chess.helpers import (
    check_significance_of_improvement,
    ChessVectorEnv,
    make_env,
    to_tensor,
)


class ChessDeepQ:
    def __init__(
        self,
        memory_size=10000,
        batch_size=32,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=1e-4,
        target_update_freq=5,
        num_envs=10,
        max_game_length=100,
        score_weight=0.5,
    ):
        self.envs = ChessVectorEnv(
            [
                make_env(max_game_length, score_weight, score_weight)
                for _ in range(num_envs)
            ]
        )
        self.num_envs = num_envs
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.episode_count = 0

        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.model = ChessCNN().to(self.device)
        self.target_model = ChessCNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def select_actions(self, states, legal_moves_masks, dones):
        """Select actions for all environments simultaneously"""
        actions = []
        states_tensor = to_tensor(states, self.device)

        with torch.no_grad():
            q_values = self.model(states_tensor)

            for i in range(self.num_envs):
                if dones[i]:
                    actions.append(0)
                    continue
                legal_indices = np.where(legal_moves_masks[i] == 1)[0]
                if random.random() < self.epsilon:
                    # Random action from legal moves
                    actions.append(np.random.choice(legal_indices))
                else:
                    # Mask illegal moves and select best legal action
                    masked_q_values = q_values[i] * torch.FloatTensor(
                        legal_moves_masks[i]
                    ).to(self.device)
                    masked_q_values[legal_moves_masks[i] == 0] = float("-inf")
                    actions.append(masked_q_values.argmax().item())

        return np.array(actions)

    def store_transitions(
        self, states, actions, rewards, next_states, dones, legal_moves_masks
    ):
        for i in range(self.num_envs):
            self.memory.append(
                (
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    dones[i],
                    legal_moves_masks[i],
                )
            )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, legal_masks = zip(*batch)

        states = to_tensor(states, self.device)
        actions = to_tensor(actions, self.device).long()
        rewards = to_tensor(rewards, self.device)
        next_states = to_tensor(next_states, self.device)
        dones = to_tensor(dones, self.device)
        legal_masks = to_tensor(legal_masks, self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss_per_sample = nn.MSELoss(reduction="none")(
            current_q_values.squeeze(), target_q_values
        )
        _legal_masks = legal_masks.gather(1, actions.unsqueeze(1)).squeeze()
        loss_per_sample = loss_per_sample.masked_fill(~_legal_masks.bool(), 10)
        loss = loss_per_sample.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train_deep_q_chess(
    episodes,
    num_envs=10,
    max_game_length=50,
    render_freq=10,
    learning_rate=5e-4,
    batch_size=64,
):

    agent = ChessDeepQ(
        num_envs=num_envs,
        max_game_length=max_game_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    best_reward = -float("inf")
    for episode in tqdm(range(1, episodes + 1)):
        states, _ = agent.envs.reset()
        episode_rewards = np.zeros(num_envs)
        episode_win_lose_draw = np.zeros(num_envs)
        losses = []
        dones = [False] * num_envs
        while not all(dones):
            legal_moves_masks = agent.envs.get_legal_moves_mask()
            actions = agent.select_actions(states, legal_moves_masks, dones)
            next_states, rewards, new_dones, _, info = agent.envs.step(actions)
            win_lose_draw = info.get("result", [None] * num_envs)

            agent.store_transitions(
                states, actions, rewards, next_states, new_dones, legal_moves_masks
            )
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            episode_rewards += rewards
            states = next_states

            for i in range(num_envs):
                if new_dones[i]:
                    if win_lose_draw[i] is not None:
                        episode_win_lose_draw[i] = win_lose_draw[i]
                    dones[i] = True

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        agent.episode_count += 1
        if agent.episode_count % agent.target_update_freq == 0:
            agent.update_target_network()

        avg_reward = np.mean(episode_rewards)
        # Max game length is 50, timestep reward is -0.1.
        num_positive = sum((episode_rewards + 5) > 0)

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.model.state_dict(), "best_parallel_chess_model.pt")

        if episode % render_freq == 0:
            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode {episode}/{episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Positive Reward Pct: {num_positive / num_envs:.2f}")
            print(f"Best Average Reward: {best_reward:.2f}")
            print(
                f"Checkmates: {sum((episode_win_lose_draw == 1)):.2f} Losses: {sum((episode_win_lose_draw == -1)):.2f} Draws: {sum((episode_win_lose_draw == 0)):.2f}"
            )
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("-" * 50)

    return agent


if __name__ == "__main__":
    num_cpus = os.cpu_count()
    print(f"Using {num_cpus} CPUs")
    stats = train_deep_q_chess(2000, num_envs=num_cpus)

from enum import StrEnum
import numpy as np
from blackjack.algorithms.base_agent import BaseAgent
from blackjack.environment import BlackjackEnv


class VisitModes(StrEnum):
    FIRST_VISIT = "first_visit"
    EVERY_VISIT = "every_visit"


class MonteCarloAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_factor=0.95,
        episode_length=100,
        mode: VisitModes = VisitModes.FIRST_VISIT,
        epsilon=0.1,
        epsilon_decay=0.99,
    ):
        super().__init__(env, discount_factor)
        self.episode_length = episode_length
        self.q_values = {}  # (state, action) -> running average
        self.n_visits = {}  # (state, action) -> count of visits
        self.policy = np.zeros((env.max_cars + 1, env.max_cars + 1), dtype=int)
        self.mode = mode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor

    def select_action(self, state):
        valid_actions = self.env.get_valid_actions(state)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = [self.q_values.get((state, a), 0.0) for a in valid_actions]
        return valid_actions[np.argmax(q_values)]

    def run_episode(self):
        episode = []
        state = self.env.reset(random_state=True)
        episode_return = 0
        for t in range(self.episode_length):
            action = self.select_action(state)
            next_state, reward = self.env.sample_step(state, action)
            if next_state is None:
                print("Invalid transition!")
                break
            episode.append((state, action, reward))
            # print(f"Step {t}: {state} --{action}--> {next_state}, R={reward}")
            episode_return += reward
            state = next_state
        return episode_return, episode

    def train(self, n_episodes=2000):
        best_return = 0
        for episode_num in range(n_episodes):
            if episode_num % 100 == 0:
                self.epsilon = self.epsilon * self.epsilon_decay
            episode_return, episode = self.run_episode()
            G = 0
            visited = set()
            t = len(episode) - 1
            while t >= 0:
                state, action, reward = episode[t]
                t -= 1
                G = reward + self.discount_factor * G
                key = (state, action)
                if self.mode == VisitModes.FIRST_VISIT and key in visited:
                    continue

                if key not in self.n_visits:
                    self.n_visits[key] = 0
                    self.q_values[key] = 0

                self.n_visits[key] += 1
                self.q_values[key] += (G - self.q_values[key]) / self.n_visits[key]

                visited.add(key)

            if episode_return > best_return:
                best_return = episode_return
            # Update policy
            for i in range(self.env.max_cars + 1):
                for j in range(self.env.max_cars + 1):
                    state = (i, j)
                    valid_actions = self.env.get_valid_actions(state)
                    q_values = [self.q_values.get((state, a), 0) for a in valid_actions]
                    self.policy[state] = valid_actions[np.argmax(q_values)]

            if episode_num % 10 == 0:
                print(f"Episode {episode_num}, Best Return: {best_return:.2f}")


if __name__ == "__main__":
    env = BlackjackEnv()
    agent = MonteCarloAgent(env)
    agent.train(n_episodes=100000)
    agent.save_policy("car_rental_policy.npy")
    agent.plot_value_function(show=True, save_path="car_rental_value_function.png")
    agent.plot_policy(show=True, save_path="car_rental_policy.png")

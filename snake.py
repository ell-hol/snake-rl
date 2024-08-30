# Snake game
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run the Snake game with RL")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--run", action="store_true", help="Run the model in inference mode"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/snake_dqn_model.pth",
        help="Path to the weights file",
    )
    return parser.parse_args()


# Fix for IndexError: list index out of range in the reset method
class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game RL")

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

        self.snake_block = 20
        self.real_time_speed = 5000
        self.slow_speed = 30
        self.is_real_time = True
        self.snake_speed = (
            self.real_time_speed if self.is_real_time else self.slow_speed
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)

        self.max_episode_steps = 800  # Maximum number of steps per episode
        self.best_score = 0  # Initialize best score
        self.reset()

    def reset(self):
        self.game_over = False
        self.x1 = self.width / 2
        self.y1 = self.height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.snake_list = [
            (self.x1, self.y1)
        ]  # Initialize snake_list with the starting position
        self.length_of_snake = 1
        self.score = 0
        self.actions_taken = 0  # Initialize actions taken
        self.accumulated_reward = 0  # Initialize accumulated reward
        self.start_time = time.time()  # Initialize start time
        self.last_actions = deque(maxlen=5)  # Store last 5 actions
        self.steps = 0  # Initialize step counter

        self.foodx = (
            round(random.randrange(0, self.width - self.snake_block) / 20.0) * 20.0
        )
        self.foody = (
            round(random.randrange(0, self.height - self.snake_block) / 20.0) * 20.0
        )

        return self._get_state()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_real_time = not self.is_real_time
                    self.snake_speed = (
                        self.real_time_speed if self.is_real_time else self.slow_speed
                    )

        # 0: LEFT, 1: RIGHT, 2: UP, 3: DOWN
        if action == 0:
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif action == 1:
            self.x1_change = self.snake_block
            self.y1_change = 0
        elif action == 2:
            self.y1_change = -self.snake_block
            self.x1_change = 0
        elif action == 3:
            self.y1_change = self.snake_block
            self.x1_change = 0

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if (
            self.x1 >= self.width
            or self.x1 < 0
            or self.y1 >= self.height
            or self.y1 < 0
        ):
            self.game_over = True

        self.window.fill(self.BLACK)
        pygame.draw.rect(
            self.window,
            self.RED,
            [self.foodx, self.foody, self.snake_block, self.snake_block],
        )

        snake_head = [self.x1, self.y1]
        self.snake_list.append(snake_head)

        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]

        for x in self.snake_list[:-1]:
            if x == snake_head:
                self.game_over = True

        self._draw_snake()
        self._display_score()
        pygame.display.update()

        reward = -1.5  # Penalize each action taken
        self.actions_taken += 1  # Increment actions taken
        self.steps += 1  # Increment step counter

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = (
                round(random.randrange(0, self.width - self.snake_block) / 20.0) * 20.0
            )
            self.foody = (
                round(random.randrange(0, self.height - self.snake_block) / 20.0) * 20.0
            )
            self.length_of_snake += 1
            self.score += 1
            reward = 100
            self.steps = 0  # Reset steps when food is eaten
            if self.score > self.best_score:
                self.best_score = self.score
        elif self.game_over:
            reward = -100  # High penalty for game over

        self.accumulated_reward += reward  # Update accumulated reward
        self.clock.tick(self.snake_speed)

        # Check if the episode has taken too long
        if self.steps >= self.max_episode_steps:
            self.game_over = True
            reward = -100  # Penalize for taking too long

        return (
            self._get_state(),
            reward,
            self.game_over,
            {"score": self.score, "accumulated_reward": self.accumulated_reward},
        )

    def _draw_snake(self):
        for x in self.snake_list:
            pygame.draw.rect(
                self.window,
                self.GREEN,
                [x[0], x[1], self.snake_block, self.snake_block],
            )

    def _display_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.window.blit(score_text, [10, 10])
        best_score_text = self.font.render(
            f"Best Score: {self.best_score}", True, self.WHITE
        )
        self.window.blit(best_score_text, [10, 40])
        reward_text = self.font.render(
            f"Reward: {self.accumulated_reward:.2f}", True, self.WHITE
        )
        self.window.blit(reward_text, [10, 70])
        speed_text = self.font.render(
            f"Speed: {'Real-time' if self.is_real_time else 'Slow'}", True, self.WHITE
        )
        self.window.blit(speed_text, [10, 100])
        time_text = self.font.render(
            f"Time: {time.time() - self.start_time:.2f}", True, self.WHITE
        )
        self.window.blit(time_text, [10, 130])
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.WHITE)
        self.window.blit(steps_text, [10, 160])

    def _get_state(self):
        head_x, head_y = self.snake_list[-1]
        state = [
            # Danger straight
            (self.x1_change == 20 and self._is_collision(head_x + 20, head_y))
            or (self.x1_change == -20 and self._is_collision(head_x - 20, head_y))
            or (self.y1_change == -20 and self._is_collision(head_x, head_y - 20))
            or (self.y1_change == 20 and self._is_collision(head_x, head_y + 20)),
            # Danger right
            (self.x1_change == 20 and self._is_collision(head_x, head_y - 20))
            or (self.x1_change == -20 and self._is_collision(head_x, head_y + 20))
            or (self.y1_change == -20 and self._is_collision(head_x + 20, head_y))
            or (self.y1_change == 20 and self._is_collision(head_x - 20, head_y)),
            # Danger left
            (self.x1_change == 20 and self._is_collision(head_x, head_y + 20))
            or (self.x1_change == -20 and self._is_collision(head_x, head_y - 20))
            or (self.y1_change == -20 and self._is_collision(head_x - 20, head_y))
            or (self.y1_change == 20 and self._is_collision(head_x + 20, head_y)),
            # Move direction
            self.x1_change == -20,  # left
            self.x1_change == 20,  # right
            self.y1_change == -20,  # up
            self.y1_change == 20,  # down
            # Food location
            self.foodx < self.x1,  # food left
            self.foodx > self.x1,  # food right
            self.foody < self.y1,  # food up
            self.foody > self.y1,  # food down
            # Time elapsed
            time.time() - self.start_time,
        ]
        return np.array(state, dtype=float)

    def _is_collision(self, x, y):
        if x >= self.width or x < 0 or y >= self.height or y < 0:
            return True
        for segment in self.snake_list[:-1]:
            if x == segment[0] and y == segment[1]:
                return True
        return False

    def render(self):
        pygame.display.update()

    def close(self):
        pygame.quit()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


class RNNAgent:
    def __init__(self, state_size, action_size, hidden_size=512, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size

        self.model = RNN(state_size, hidden_size, action_size).to(self.device)
        self.target_model = RNN(state_size, hidden_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)

        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

    def act(self, state, inference=False):
        if not inference and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values, _ = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.unsqueeze(1).to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.unsqueeze(1).to(self.device)
        done = done.to(self.device)

        q_values, _ = self.model(state)
        q_values = q_values.gather(1, action.unsqueeze(1))

        next_q_values, _ = self.target_model(next_state)
        next_q_values = next_q_values.max(1)[0].detach()
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train_rnn(
    episodes, render=False, checkpoint_interval=50, checkpoint_dir="checkpoints"
):
    env = SnakeEnv()
    state_size = 12  # Size of the state returned by SnakeEnv (increased by 1 for time)
    action_size = 4  # Number of possible actions
    agent = RNNAgent(state_size, action_size)

    scores = []
    accumulated_rewards = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for e in range(episodes):
        state = env.reset()
        score = 0
        accumulated_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            score += reward
            accumulated_reward = info["accumulated_reward"]
            agent.train()

            if render:
                env.render()
                episode_text = env.font.render(
                    f"Episode: {e+1}/{episodes}", True, env.WHITE
                )
                env.window.blit(episode_text, [10, 190])
                pygame.display.update()

        if e % agent.target_update == 0:
            agent.update_target_model()

        scores.append(score)
        accumulated_rewards.append(accumulated_reward)
        print(
            f"Episode: {e+1}/{episodes}, Score: {score:.2f}, Accumulated Reward: {accumulated_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

        # Save checkpoint
        if (e + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{e+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {e+1}")

    # Save the final trained model
    torch.save(
        agent.model.state_dict(), os.path.join(checkpoint_dir, "snake_dqn_model.pth")
    )

    env.close()
    return scores, accumulated_rewards, agent


if __name__ == "__main__":

    args = parse_args()  # Parse the command-line arguments

    if args.train:

        # Train the model
        episodes = 1000
        scores, accumulated_rewards, agent = train_rnn(episodes, render=True)

        # Plot the results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(scores)), scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("DQN Training Progress - Scores")

        plt.subplot(1, 2, 2)
        plt.plot(range(len(accumulated_rewards)), accumulated_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")
        plt.title("DQN Training Progress - Accumulated Rewards")

        plt.tight_layout()
        plt.show()

    elif args.run:
        state_size = (
            12  # Size of the state returned by SnakeEnv (increased by 1 for time)
        )
        action_size = 4  # Number of possible actions
        agent = RNNAgent(state_size, action_size)
        # load the agent
        agent.model.load_state_dict(torch.load(args.weights))
        agent.model.eval()

        # Run the game
        env = SnakeEnv()
        state = env.reset()
        done = False

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_q
                    ):
                        env.close()
                        pygame.quit()
                        exit()

                action = agent.act(state, inference=True)
                next_state, reward, done, info = env.step(action)
                state = next_state
                env.render()

                if done:
                    state = env.reset()
                    done = False

        except KeyboardInterrupt:
            env.close()

import gymnasium as gym
import torch.nn as nn
import torch
from deep_q_learning_torch import DQN
import numpy as np


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.Linear(state_space_shape, 32),
        nn.ReLU(),
        nn.Linear(32, num_actions),
    )


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode=None).env

    state_space_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n
    model = build_model(state_space_shape, num_actions)

    num_episodes = 10000
    num_steps_per_episode = 1000
    epsilon = 1
    train_reward = 0
    train_steps = 0


    agent = DQN(state_space_shape, num_actions, model, model)

    for episode in range(num_episodes):
        state, _ = env.reset()

        # env.render()
        for step in range(num_steps_per_episode):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_memory(state, action, reward, next_state, done)

            train_reward += reward
            train_steps += 1

            if done:
                break

            state = next_state

        if epsilon >= 0.1:
            epsilon *= 0.99

        agent.train()

        if episode % 10:
            agent.update_target_model()

    print(f"train reward: {train_reward/10000} train steps: {train_steps/10000}")
    total_reward = 0
    total_steps = 0
    for iteration in range(100):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < num_steps_per_episode:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent.model(state)).item()
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            total_reward += reward
            steps += 1
            total_steps += 1

            state = next_state

        if iteration == 49:
            print(f"10000 Episodes {total_reward/50} steps: {total_steps/50}")

    print(f"10000 Episodes {total_reward/100} steps: {total_steps/100}")


    num_episodes = 30000
    epsilon = 1
    train_steps = 0
    train_reward = 0

    agent = DQN(state_space_shape, num_actions, model, model)

    for episode in range(num_episodes):
        state, _ = env.reset()

        # env.render()
        for step in range(num_steps_per_episode):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            train_reward += reward
            train_steps += 1

            agent.update_memory(state, action, reward, next_state, done)

            if done:
                break

            state = next_state

        if epsilon >= 0.1:
            epsilon *= 0.99

        agent.train()

        if episode % 10:
            agent.update_target_model()

    print(f"30000 ep train reward {train_reward/30000} train steps {train_steps/30000}")

    total_reward = 0
    total_steps = 0
    for iteration in range(100):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < num_steps_per_episode:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action = torch.argmax(agent.model(state)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            total_reward += reward
            steps += 1
            total_steps += 1

            state = next_state

        if iteration == 49:
            print(f"30000 ep reward {total_reward/50} steps: {total_steps/50}")

    print(f"30000 ep reward {total_reward/100} steps: {total_steps/100}")










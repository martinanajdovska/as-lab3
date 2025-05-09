from collections import deque

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from deep_q_learning_torch import DuelingDQN
import ale_py
import shimmy


def preprocess_state(frame, new_width=84, new_height=84):
    image = Image.fromarray(frame)
    image = image.resize((new_width, new_height))
    image = image.convert('L')
    processed_frame = np.array(image) / 255.0

    return processed_frame


def preprocess_reward(reward):
    return np.clip(reward, -1000.0, 1000.0)


if __name__ == '__main__':
    device = "cuda"
    env = gym.make('ALE/MsPacman-v5', render_mode=None)

    state_space_shape = (1, env.observation_space.shape[0], env.observation_space.shape[1])
    num_actions = env.action_space.n

    agent = DuelingDQN(state_space_shape=state_space_shape, num_actions=num_actions)

    num_episodes = 5000
    episodes_length = 1000
    epsilon = 1
    train_reward = 0
    train_steps = 0

    for episode in range(num_episodes):
        frame, info = env.reset()
        frame = preprocess_state(frame)
        done = False
        steps = 0

        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        state = torch.from_numpy(state).float().to(device)

        while not done and steps < episodes_length:
            action = agent.get_action(state, epsilon)

            next_frame, reward, terminated, _, _ = env.step(action)
            reward = preprocess_reward(reward)
            train_reward += reward

            next_frame = preprocess_state(next_frame)
            stacked_frames.append(next_frame)
            next_state = np.stack(list(stacked_frames), axis=0)
            next_state = torch.from_numpy(next_state).float().to(device)

            agent.update_memory(state, action, reward, next_state, terminated)

            state = next_state
            steps += 1
            train_steps += 1

        if epsilon >= 0.1:
            epsilon *= 0.99

        agent.train()

        if episode % 10:
            agent.update_target_model()

    print(f"train reward: {train_reward / 5000} train steps: {train_steps / 5000}")

    total_reward = 0
    total_steps = 0
    for iteration in range(100):
        frame, info = env.reset()
        frame = preprocess_state(frame)
        done = False
        steps = 0

        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        state = torch.from_numpy(state).float().to(device)

        while not done and steps < episodes_length:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(agent.model(state)).item()

            next_frame, reward, terminated, _, _ = env.step(action)
            reward = preprocess_reward(reward)
            total_reward += reward

            next_frame = preprocess_state(next_frame)
            stacked_frames.append(next_frame)
            next_state = np.stack(list(stacked_frames), axis=0)
            next_state = torch.from_numpy(next_state).float().to(device)

            agent.update_memory(state, action, reward, next_state, terminated)

            state = next_state
            steps += 1
            total_steps += 1

        if iteration == 49:
            print(f"5000 Episodes reward {total_reward / 50} steps: {total_steps / 50}")

    print(f"5000 Episodes reward {total_reward / 100} steps: {total_steps / 100}")

    agent = DuelingDQN(state_space_shape=state_space_shape, num_actions=num_actions)

    num_episodes = 20000
    epsilon = 1
    train_reward = 0
    train_steps = 0

    for episode in range(num_episodes):
        frame, info = env.reset()
        frame = preprocess_state(frame)
        done = False
        steps = 0

        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        state = torch.from_numpy(state).float().to(device)

        while not done and steps < episodes_length:
            action = agent.get_action(state, epsilon)

            next_frame, reward, terminated, _, _ = env.step(action)
            reward = preprocess_reward(reward)
            train_reward += reward

            next_frame = preprocess_state(next_frame)
            stacked_frames.append(next_frame)
            next_state = np.stack(list(stacked_frames), axis=0)
            next_state = torch.from_numpy(next_state).float().to(device)

            agent.update_memory(state, action, reward, next_state, terminated)

            state = next_state
            steps += 1
            train_steps += 1

        if epsilon >= 0.1:
            epsilon *= 0.99

        agent.train()

        if episode % 10:
            agent.update_target_model()

    print(f"train reward: {train_reward / 20000} train steps: {train_steps / 20000}")

    total_reward = 0
    total_steps = 0
    for iteration in range(100):
        frame, info = env.reset()
        frame = preprocess_state(frame)
        done = False
        steps = 0

        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
        state = np.stack(stacked_frames, axis=0)
        state = torch.from_numpy(state).float()

        while not done and steps < episodes_length:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(agent.model(state)).item()

            next_frame, reward, terminated, _, _ = env.step(action)
            reward = preprocess_reward(reward)
            total_reward += reward

            next_frame = preprocess_state(next_frame)
            stacked_frames.append(next_frame)
            next_state = np.stack(list(stacked_frames), axis=0)
            next_state = torch.from_numpy(next_state).float().to(device)

            agent.update_memory(state, action, reward, next_state, terminated)

            state = next_state
            steps += 1
            total_steps += 1

        if iteration == 49:
            print(f"20000 Episodes reward {total_reward / 50} steps: {total_steps / 50}")

    print(f"20000 Episodes reward {total_reward / 100} steps: {total_steps / 100}")

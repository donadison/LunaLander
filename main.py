import csv
import sys
import os
import pygame
import numpy as np
import torch

import DQNAgent
import MoonLanderEnv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up the agent
    state_size = 5  # state vector length
    episodes = 1000
    batch_size = 32
    steps_per_episode = 500
    clock = pygame.time.Clock()
    env = MoonLanderEnv.MoonLanderEnv()
    agent = DQNAgent.DQNAgent(
            state_size=5,
            action_size=4,
            device=device,
            epsilon_start=1.0,  # Start with full exploration
            epsilon_end=0.01,  # Minimum exploration rate
            epsilon_decay_rate=0.999,  # Very slow decay for long training
            max_steps=episodes  # Very slow decay for long training
        )

    for _ in range(5):
        if not os.path.exists('rewards.csv'):
            with open('rewards.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward"])
            last_episode = 0
        else:
            with open('rewards.csv', 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                if len(rows) > 1:
                    last_episode = int(rows[-1][0])
                else:
                    last_episode = 0

        # Load saved memory if exists
        if os.path.exists("replay_memory.pkl"):
            print("Loading replay memory...")
            agent.load_memory("replay_memory.pkl")

        # Load saved model if exists
        if os.path.exists("lander_pytorch.pth"):
            print("Loading saved model...")
            agent.load("lander_pytorch.pth")

        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            for _ in range(steps_per_episode):
                action = agent.act(state, e)
                next_state, reward, reason, done = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(
                        f"Episode {e + 1}/{episodes} - Reward: {reward:.2f} - Reason: {reason} - Epsilon: {agent.epsilon:.2f}")
                    with open('rewards.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([last_episode + e + 1, reward])
                    break

            if len(agent.memory) > batch_size:  # Replay every 10 steps
                agent.replay(batch_size)

        print("Saving replay memory")
        agent.save_memory("replay_memory.pkl")
        print("Saving trained model...")
        agent.save('lander_pytorch.pth')

    # Inicjalizacja Pygame
    pygame.init()
    pygame.display.set_mode((600, 400), flags=pygame.SHOWN)
    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.plot.plot_rewards()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, reason, done = env.step(action)

            env.render()
            state = np.reshape(next_state, [1, state_size])

            clock.tick(30)  # Limit to 30 FPS

            if done:
                print(f"Episode complete! Reward: {reward} - Reason: {reason}")
                break

if __name__ == "__main__":
    main()

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

    # Load saved model if exists
    if os.path.exists("lander_pytorch.pth"):
        print("Loading saved model...")
        agent.load("lander_pytorch.pth")

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

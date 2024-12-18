import os
import re
import sys
import csv
import pygame
import torch

import MoonLanderEnv
import DQNAgent

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up the agent
    state_size = 5  # state vector length
    env = MoonLanderEnv.MoonLanderEnv()
    agent = DQNAgent.DQNAgent(
        state_size=state_size,
        action_size=4,
        device=device,
        epsilon_start=0.5,  # Start with half-full exploration
        epsilon_end=0.05,  # Minimum exploration rate
        epsilon_decay_rate=0.99,  # Very slow decay for long training
        episode_num=1000  # Placeholder, not used during testing
    )

    outcome = {
        0: "Brak powodu",
        1: "Howdy, Houston! Lądownik zaparkowany prosto jak w saloonie.",
        2: "Zwolnij kowboju! Platforma ledwo ustała!",
        3: "Krzywo jak dach stodoły po burzy, ale się trzyma!",
        4: "Za szybko i za krzywo – lądowanie jak u pijącego szeryfa w miasteczku!",
        5: "Kowboj przesadził i poleciał prosto w gwiazdy! Orbitę opuścił szybciej niż pędzący meteoryt.",
        6: "Dziki zachód wymaga dzikiej precyzji, kowboju!",
        7: "Kowboj wpadł na skały jak kaktus w burzę piaskową!",
        8: "Czas się skończył, kowboju! Nie wykonano misji na czas."
    }

    # Find all model files
    model_files = [f for f in os.listdir('.') if re.match(r'lander_pytorch_(\d+)\.pth', f)]
    model_files.sort(key=lambda x: int(re.search(r'lander_pytorch_(\d+)\.pth', x).group(1)))

    if not model_files:
        print("No saved models found.")
        return

    # Open CSV file for writing results
    with open('../../model_statistics.csv', mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['model_num', 'rep_num', 'reward', 'reason'])

    # Test each model 10,000 times
    for model_file in model_files:
        model_num = int(re.search(r'lander_pytorch_(\d+)\.pth', model_file).group(1))
        print(f"Testing model: {model_file}")
        agent.load(model_file)

        for rep in range(1000):
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, reason, done = env.step(action)
                state = next_state

                if done:
                    # Log the result
                    with open('../../model_statistics.csv', mode='a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([model_num, rep + 1, reward, reason])
                    break

if __name__ == "__main__":
    main()

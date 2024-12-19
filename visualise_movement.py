import csv
import torch
import matplotlib.pyplot as plt
import random

import MoonLanderEnv
import DQNAgent


def de_preprocess_state(state):
    state[0] = (state[0] + 1) * 300  # Denormalize X position (-1 to 1 -> 0 to 600)
    state[1] = (state[1] + 1) * 200  # Denormalize Y position (-1 to 1 -> 0 to 400)
    state[2] = state[2] * 20  # Denormalize X velocity (-1 to 1 -> -20 to 20)
    state[3] = state[3] * 15  # Denormalize Y velocity (-1 to 1 -> -15 to 15)
    state[4] = state[4] * 360  # Denormalize angle (-1 to 1 -> -360 to 360)
    return state


def record_states():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up the agent
    state_size = 5  # State vector length
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

    # Load pre-trained model
    agent.load('lander_pytorch_245000.pth')

    # Open a CSV file to store positions for every step
    with open('visualise_245000.csv', mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Header for the CSV
        csv_writer.writerow(['Episode', 'Step', 'Xpos', 'Ypos', 'Xvel', 'Yvel', 'Angle'])

    # Iterate through 1000 episodes
    for episode_num in range(1000):
        state = env.reset()
        done = False
        step_num = 0

        # Run the episode until termination
        while not done:
            # Choose the action and step the environment
            action = agent.act(state)
            next_state, reward, reason, done = env.step(action)

            # De-process the next state for storing
            deprocessed_state = de_preprocess_state(next_state.copy())

            # Write the position and other state values to the CSV
            with open('visualise_245000.csv', mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    episode_num + 1,  # Current episode number
                    step_num + 1,    # Current step number
                    deprocessed_state[0],  # Deprocessed X position
                    deprocessed_state[1],  # Deprocessed Y position
                    deprocessed_state[2],  # Deprocessed X velocity
                    deprocessed_state[3],  # Deprocessed Y velocity
                    deprocessed_state[4],  # Deprocessed angle
                ])

            # Update state and step number
            state = next_state
            step_num += 1

            if done:
                break


def plot_trajectories(csv_file):
    x_positions = []
    y_positions = []
    episode_numbers = set()  # To track unique episodes

    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            episode_num, step, xpos, ypos, _, _, _ = row
            xpos, ypos = float(xpos), float(ypos)
            x_positions.append(xpos)
            y_positions.append(ypos)
            episode_numbers.add(int(episode_num))

    plt.figure(figsize=(10, 6))

    random_episodes = random.sample(sorted(episode_numbers), 100)  # Randomly select 100 unique episodes

    for episode_num in random_episodes:
        episode_x = []
        episode_y = []

        with open(csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row

            for row in csv_reader:
                ep_num, step, xpos, ypos, _, _, _ = row
                if int(ep_num) == episode_num:
                    episode_x.append(float(xpos))
                    episode_y.append(float(ypos))

        plt.plot(episode_x, episode_y)

    plt.gca().invert_yaxis()
    plt.title('Landing Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()


plot_trajectories('visualise_245000.csv')

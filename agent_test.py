import os
import re
import sys
import pygame
import torch

import MoonLanderEnv
import DQNAgent


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')  # Set device based on availability (GPU or CPU)
    print('Using device:', device)

    # Define state size and number of episodes
    state_size = 5  # state vector length
    episodes = 1000  # Number of episodes to run, unused in testing
    clock = pygame.time.Clock()  # Pygame clock for controlling frame rate
    env = MoonLanderEnv.MoonLanderEnv()  # Initialize the custom Moon Lander environment
    agent = DQNAgent.DQNAgent(  # Instantiate the DQN agent
        state_size=state_size,  # Size of the state space
        action_size=4,  # Number of possible actions
        device=device,  # Device to use for computations
        epsilon_start=0.5,  # Exploration-exploitation trade-off parameter (start high), unused in testing
        epsilon_end=0.05,  # Minimum exploration probability, unused in testing
        epsilon_decay_rate=0.99,  # Slow decay of epsilon for long training, unused in testing
        episode_num=episodes  # Total number of episodes per epsilon span, unused in testing
    )

    # Define possible outcomes and reasons for episode termination
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

    # Find the model file with the highest number
    model_path = None
    model_files = [f for f in os.listdir('.') if
                   re.match(r'lander_pytorch_(\d+)\.pth', f)]  # Find model files matching pattern
    if model_files:
        # Extract numbers and find the highest
        model_files.sort(key=lambda x: int(re.search(r'lander_pytorch_(\d+)\.pth', x).group(1)),
                         reverse=True)  # Sort by number
        model_path = model_files[0]  # Use the latest model
        print(f"Loading saved model: {model_path}")
        agent.load(model_path)  # Load the pre-trained model
    else:
        print("No saved model found. Starting with a new agent.")
    agent.load(
        'lander_pytorch_100000.pth')  # Ensure loading a specific model, overwrites automatic highest model selection

    # Initialize Pygame
    pygame.init()  # Initialize Pygame for rendering
    pygame.display.set_mode((600, 400), flags=pygame.SHOWN)  # Set display mode for Pygame window

    while True:
        state = env.reset()  # Start a new episode and get the initial state
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Check for quit event
                    pygame.quit()  # Exit Pygame
                    env.plot.plot_rewards()  # Plot the training rewards per episode
                    sys.exit()  # Exit the script

            # Use the trained model to select an action
            action = agent.act(state)

            # Step the environment with the selected action
            next_state, reward, reason, done = env.step(action)

            # Move to the next state
            state = next_state

            # Render the environment for visualization
            env.render()

            # Limit to 30 FPS
            clock.tick(30)

            if done:
                outcome_string = outcome.get(reason, "Nieznany powód")  # Get the outcome string
                print(f"Episode complete! Reward: {reward} - Reason: {outcome_string}")
                break


if __name__ == "__main__":
    main()

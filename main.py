import csv
import os
import torch

import DQNAgent
import MoonLanderEnv


def main():
    # Check if a GPU is available, otherwise use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Define training parameters
    state_size = 5  # Number of features in the state vector
    episodes = 30000  # Total number of episodes for training
    batch_size = 64  # Number of experiences sampled per training step
    steps_per_episode = 500  # Maximum steps per episode

    # Initialize the environment and agent
    env = MoonLanderEnv.MoonLanderEnv()
    agent = DQNAgent.DQNAgent(
        state_size=5,  # Size of state vector
        action_size=4,  # Number of possible actions
        device=device,  # Device (CPU or GPU) for computations
        epsilon_start=0.75,  # Initial exploration rate
        epsilon_end=0.01,  # Minimum exploration rate
        epsilon_decay_rate=0.99,  # Slow decay for exploration rate
        episode_num=episodes,  # Total number of episodes
        memory_capacity=episodes * steps_per_episode // 5,  # Memory size for experience replay
        target_update_interval=10  # Frequency of target model updates
    )

    # Mapping of outcome reasons to descriptive messages
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

    while True:
        # Load the latest model if it exists
        model_files = [f for f in os.listdir('.') if f.startswith('lander_pytorch_') and f.endswith('.pth')]
        if model_files:
            # Sort and select the latest model by episode number
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = model_files[-1]
            print(f"Loading the latest model: {latest_model}")
            agent.load(latest_model)
            agent.epsilon_start = 0.33  # Reduce exploration for resumed training
            agent.epsilon_end = 0.01  # Minimum exploration rate
            last_episode = int(latest_model.split('_')[-1].split('.')[0])  # Get the last episode number
        else:
            # Start training from scratch if no models are found
            print("No existing models found, starting from scratch.")
            last_episode = 0

        # Create rewards file if it doesn't exist
        if not os.path.exists('../../rewards.csv'):
            with open('../../rewards.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Outcome"])

        # Load replay memory if it exists
        if os.path.exists("replay_memory.pkl"):
            print("Loading replay memory...")
            agent.load_memory("replay_memory.pkl")

        # Train for the specified number of episodes
        for e in range(episodes):
            state = env.reset()  # Reset environment at the start of each episode

            for _ in range(steps_per_episode):
                # Choose an action based on the current state
                action = agent.act(state, e)
                # Execute the action in the environment
                next_state, reward, reason, done = env.step(action)

                # Store the experience in replay memory
                agent.remember(state, action, reward, next_state, done)
                state = next_state  # Update the current state

                if done:
                    # Log the episode result
                    outcome_string = outcome.get(reason, "Nieznany powód")  # Get outcome message
                    print(
                        f"Episode {last_episode + e + 1}/{last_episode + episodes} - Reward: {reward:.2f} - Reason: {outcome_string} - Epsilon: {agent.epsilon:.2f}")
                    # Save rewards and outcomes to a CSV file
                    with open('rewards.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([last_episode + e + 1, reward, reason])
                    break

            # Train the model if enough experiences are collected
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Save the model periodically
            if (e + 1) % 1000 == 0:
                model_filename = f'lander_pytorch_{last_episode + e + 1}.pth'
                print(f"Saving trained model as {model_filename}")
                agent.save(model_filename)

            # Save the replay memory periodically
            if (e + 1) % 10000 == 0:
                print("Saving replay memory")
                agent.save_memory("replay_memory.pkl")


if __name__ == "__main__":
    main()

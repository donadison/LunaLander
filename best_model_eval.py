import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt

import MoonLanderEnv
import DQNAgent


def run_model():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up the agent
    state_size = 5  # state vector length
    env = MoonLanderEnv.MoonLanderEnv()  # Initialize the Moon Lander environment
    agent = DQNAgent.DQNAgent(  # Initialize the DQN agent
        state_size=state_size,
        action_size=4,  # Number of possible actions
        device=device,  # Device (GPU/CPU)
        epsilon_start=0.5,  # Start with half-full exploration, not used during testing
        epsilon_end=0.05,  # Minimum exploration rate, not used during testing
        epsilon_decay_rate=0.99,  # Very slow decay for long training, not used during testing
        episode_num=1000  # Placeholder, not used during testing
    )

    # Load pre-trained model
    agent.load('lander_pytorch_100000.pth')

    # Open CSV file for writing results
    with open('model_eval.csv', mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['rep_num', 'reason'])  # Write header row

    for rep in range(10000):  # Run the model for 10,000 episodes
        state = env.reset()  # Reset the environment to get the initial state
        done = False  # Flag indicating whether the episode is done

        while not done:
            action = agent.act(state)  # Select action based on the agent's policy
            next_state, reward, reason, done = env.step(action)  # Take step in environment
            state = next_state  # Update the state

            if done:
                # Record the outcome reason to the CSV file
                with open('model_eval.csv', mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([rep + 1, reason])
                break


def plot_histogram():
    model_name = 'model_eval.csv'
    data = pd.read_csv(model_name)  # Load the evaluation results from CSV

    # Mapping of reason codes to their descriptions
    outcome = {
        0: "Brak powodu",
        1: "Perfekcyjne lądowanie",
        2: "Platforma ledwo ustała",
        3: "Krzywo jak dach",
        4: "Za szybko i za krzywo",
        5: "Poleciał prosto w gwiazdy",
        6: "Dziki zachód wymaga precyzji",
        7: "Wpadł na skały",
        8: "Czas się skończył",
    }

    # Mapping of reason codes to colors for visualization
    outcome_colors = {
        0: 'gray',  # "Brak powodu"
        1: 'green',  # "Perfekcyjne lądowanie"
        2: 'yellow',  # "Platforma ledwo ustała"
        3: 'orange',  # "Krzywo jak dach"
        4: 'red',  # "Za szybko i za krzywo"
        5: 'purple',  # "Poleciał prosto w gwiazdy"
        6: 'brown',  # "Dziki zachód wymaga precyzji"
        7: 'black',  # "Wpadł na skały"
        8: 'blue'  # "Czas się skończył"
    }

    reason_counts = data['reason'].value_counts().sort_index()  # Count occurrences of each reason
    total_counts = reason_counts.sum()

    reason_percentages = (reason_counts / total_counts) * 100  # Calculate percentages

    bar_colors = [outcome_colors[reason] for reason in reason_counts.index]  # Get colors for each reason

    plt.figure(figsize=(12, 7))
    bars = plt.bar(reason_counts.index, reason_counts.values, color=bar_colors, edgecolor='black', alpha=0.7)

    # Add percentages on top of bars
    for bar, percentage in zip(bars, reason_percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{percentage:.1f}%",
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(reason_counts.index, [outcome[reason] for reason in reason_counts.index], rotation=45, fontsize=10,
               ha='right')

    plt.xlabel('Reasons for Outcome', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Histogram of Reasons for Moon Lander Outcomes with Percentages', fontsize=14)

    # Create legend with description and colors
    legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=f"{key}: {value}")
                     for key, (color, value) in
                     zip(outcome_colors.keys(), zip(outcome_colors.values(), outcome.values()))]
    plt.legend(handles=legend_labels, title="Outcome Codes", loc='upper right', fontsize=10)

    histogram_output = 'reasons_histogram_with_percentages.png'
    plt.tight_layout()
    plt.savefig(histogram_output)  # Save the histogram as an image
    plt.show()  # Display the plot

    print(f"Histogram saved as {histogram_output}")


def main():
    plot_histogram()


if __name__ == "__main__":
    main()

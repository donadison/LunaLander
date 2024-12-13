import pandas as pd
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def plot_rewards(self):
        try:
            data = pd.read_csv(self.csv_file)

            if 'Episode' not in data.columns or 'Reward' not in data.columns:
                raise ValueError("CSV file must contain 'Episode' and 'Reward' columns.")

            # Plot total rewards per episode
            plt.figure(figsize=(12, 6))
            plt.scatter(data['Episode'], data['Reward'], label='Total Reward', color='blue', s=5)

            # Calculate and plot average reward per 1000 episodes
            window_size = 1000
            if len(data) >= window_size:
                data['Avg_Reward'] = data['Reward'].rolling(window=window_size).mean()
                plt.plot(data['Episode'], data['Avg_Reward'], label=f'Average Reward (per {window_size} episodes)', color='red', linewidth=2)

            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Reward', fontsize=14)
            plt.title('Reinforcement Learning Progress: Rewards per Episode', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.show()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")
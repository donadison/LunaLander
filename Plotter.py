import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def plot_rewards(self):
        try:
            data = pd.read_csv(self.csv_file)

            if 'Episode' not in data.columns or 'Reward' not in data.columns:
                raise ValueError("CSV file must contain 'Episode' and 'Total Reward' columns.")

            plt.figure(figsize=(10, 6))
            plt.scatter(data['Episode'], data['Reward'], label='Total Reward', color='blue')

            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Total Reward', fontsize=14)
            plt.title('Reinforcement Learning Progress: Total Reward per Episode', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.show()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")


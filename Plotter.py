import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def plot_rewards(self):
        try:
            # Load the CSV data
            data = pd.read_csv(self.csv_file)

            # Check if required columns exist
            required_columns = {'Episode', 'Reward', 'Outcome'}
            if not required_columns.issubset(data.columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")

            # Define a color map for the outcomes
            outcome_colors = {
                0: 'gray',    # "Brak powodu"
                1: 'green',   # "Perfekcyjne lądowanie"
                2: 'yellow',  # "Platforma ledwo ustała"
                3: 'orange',  # "Krzywo jak dach"
                4: 'red',     # "Za szybko i za krzywo"
                5: 'purple',  # "Poleciał prosto w gwiazdy"
                6: 'brown',   # "Dziki zachód wymaga precyzji"
                7: 'black',   # "Wpadł na skały"
                8: 'blue'     # "Czas się skończył"
            }

            # Assign colors to each point based on the Outcome column
            data['Color'] = data['Outcome'].map(outcome_colors).fillna('gray')

            # Calculate occurrence count and percentage for each Outcome
            outcome_counts = data['Outcome'].value_counts().sort_index()
            total_outcomes = outcome_counts.sum()
            outcome_percentages = (outcome_counts / total_outcomes * 100).round(2)

            # Prepare legend labels with count and percentage
            legend_labels = {
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

            legend_patches = [
                plt.Line2D([0], [0], marker='o', color='w', label=f"{legend_labels[outcome]}: {outcome_counts[outcome]} ({outcome_percentages[outcome]}%)",
                           markersize=10, markerfacecolor=outcome_colors[outcome])
                for outcome in outcome_counts.index
            ]

            # Set up the plot
            plt.figure(figsize=(12, 6))

            # Scatter plot of rewards per episode with colors
            plt.scatter(data['Episode'], data['Reward'], c=data['Color'], label='Total Reward', s=2, alpha=0.7)

            # Calculate and plot rolling average of rewards
            window_size = 1000
            if len(data) >= window_size:
                data['Avg_Reward'] = data['Reward'].rolling(window=window_size).mean()
                plt.plot(data['Episode'], data['Avg_Reward'],
                         label=f'Average Reward (per {window_size} episodes)', color='cyan', linewidth=2)

            # Labels and title
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Reward', fontsize=14)
            plt.title('Reinforcement Learning Progress: Rewards per Episode', fontsize=16)

            # Add legend with counts and percentages
            plt.legend(handles=legend_patches, title="Outcome Legend", fontsize=10, title_fontsize=12, loc='best')

            # Enable fine grid lines
            plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
            plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
            plt.minorticks_on()

            # Display the plot
            plt.tight_layout()
            plt.show()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
        except Exception as e:
            print(f"An error occurred: {e}")
import pandas as pd
import matplotlib.pyplot as plt

data_file = 'model_statistics.csv'
data = pd.read_csv(data_file)

mean_rewards = data.groupby('model_num')['reward'].mean().reset_index()

best_model_num = mean_rewards['model_num'][mean_rewards['reward'].idxmax()]
best_mean_reward = mean_rewards['reward'].max()

plt.figure(figsize=(12, 7))
plt.bar(mean_rewards['model_num'], mean_rewards['reward'], color='black', edgecolor='black')
plt.scatter(best_model_num, best_mean_reward, color='red', s=10, edgecolor='black', zorder=5, label=f'Best Reward: {best_mean_reward:.2f}')

plt.xlabel('Model Number', fontsize=12)
plt.ylabel('Mean Reward', fontsize=12)
plt.title('Mean Reward of custom DQN model (Inverted y scale)', fontsize=14)

step = 10000
x_ticks = mean_rewards['model_num'][::max(1, len(mean_rewards) // (mean_rewards['model_num'].max() // step))]
plt.xticks(x_ticks, rotation=45, fontsize=10)

plt.gca().invert_yaxis()
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()

output_plot = 'mean_rewards_plot.png'
plt.savefig(output_plot)
plt.show()

print(f"Bar chart saved as {output_plot}")

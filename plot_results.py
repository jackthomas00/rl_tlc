import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the metrics.csv from one trial
metrics_path = "/app/src/runs/metrics.csv"  # update path if needed
df = pd.read_csv(metrics_path)

# Convert Episode to index (or keep on X-axis)
episodes = df["Episode"]

# Plot cumulative reward
plt.figure(figsize=(12, 6))
plt.plot(episodes, df["Cumulative Reward"], label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Progression Over Time")
plt.grid(True)
plt.legend()
plt.savefig("src/plots/cumulative_reward.png", dpi=300)
plt.show()

# Plot avg waiting time vs. episode
plt.figure(figsize=(12, 6))
plt.plot(episodes, df["Average Waiting Time"], label="Avg Waiting Time", color="orange")
plt.xlabel("Episode")
plt.ylabel("Approx Avg Waiting Time")
plt.title("Average Waiting Time Over Time")
plt.grid(True)
plt.legend()
plt.savefig("src/plots/average_waiting_time.png", dpi=300)
plt.show()

# Optional: Plot queue length, actor loss, etc.
plt.figure(figsize=(12, 6))
plt.plot(episodes, df["Queue Length"], label="Queue Length", color="green")
plt.plot(episodes, df["Actor Loss"], label="Actor Loss", linestyle="--", color="red")
plt.plot(episodes, df["Critic Loss"], label="Critic Loss", linestyle="--", color="purple")
plt.xlabel("Episode")
plt.title("Training Dynamics")
plt.grid(True)
plt.legend()
plt.savefig("src/plots/queue_actor_critic.png", dpi=300)
plt.show()

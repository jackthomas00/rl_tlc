import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

base_path = "/app/src/ray_results/traffic_control"

# Find all progress.csv files
metric_files = glob(os.path.join(base_path, "**/progress.csv"), recursive=True)
print(f"Found {len(metric_files)} progress.csv files")

# Aggregate final episode per trial
results = []
for file in metric_files:
    try:
        df = pd.read_csv(file)
        if not df.empty:
            last = df.iloc[-1]
            trial_name = file.split("/")[-3]
            results.append({
                "trial": trial_name,
                "reward": last["episode_reward"],
                "waiting": last["avg_waiting_time"],
                "speed": last["avg_speed"],
                "queue": last["queue_length"]
            })
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        continue

# Plot results
if results:
    results_df = pd.DataFrame(results)
    results_df.sort_values("waiting", inplace=True)

    # Scatter plot: Reward vs. Waiting Time
    plt.figure(figsize=(10,6))
    plt.scatter(results_df["waiting"], results_df["reward"], c='blue')
    plt.xlabel("Average Waiting Time")
    plt.ylabel("Episode Reward")
    plt.title("Trade-off: Waiting Time vs. Reward per Trial")
    plt.grid(True)
    plt.savefig(f"src/plots/tradeoff_plot.png")
else:
    print("No results were processed successfully")

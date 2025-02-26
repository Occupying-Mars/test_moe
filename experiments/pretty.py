import os
import mlflow
import pandas as pd
from tabulate import tabulate
from mlflow.tracking import MlflowClient

# Define the directory where combined logs will be saved
LOGS_DIR = "images"
OUTPUT_FILE = os.path.join(LOGS_DIR, "combined_logs.txt")

# Initialize MLflow Client
client = MlflowClient()

# Get all experiments
try:
    experiments = mlflow.search_experiments()  # Use search_experiments() for newer MLflow versions
except AttributeError:
    print("[Error] Failed to list experiments. Check your MLflow version.")
    exit(1)

# List to store extracted log data
log_data = []

# Iterate over all experiments
for experiment in experiments:
    experiment_name = experiment.name
    experiment_id = experiment.experiment_id

    # Get all runs in this experiment, sorted by start time descending
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time desc"]
    )

    if not runs.empty:
        # Use the latest run
        latest_run = runs.iloc[0]
        run_id = latest_run.run_id

        # Fetch run data
        run = client.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        # Extract required metrics
        total_training_time = metrics.get("total_training_time_s", "N/A")
        val_loss = metrics.get("val_loss", "N/A")
        train_loss = metrics.get("train_loss", "N/A")

        # Extract total_params from logged params if available
        total_params = params.get("total_parameters", "N/A")

        # Store in log_data
        log_data.append({
            "Experiment Name": experiment_name,
            "Total Training Time (s)": total_training_time,
            "Validation Loss": val_loss,
            "Training Loss": train_loss,
            "Total Params": total_params
        })
    else:
        # No runs found for this experiment
        log_data.append({
            "Experiment Name": experiment_name,
            "Total Training Time (s)": "N/A",
            "Validation Loss": "N/A",
            "Training Loss": "N/A",
            "Total Params": "N/A"
        })

# Convert log data to a Pandas DataFrame
df = pd.DataFrame(log_data)

# Save as a pretty table using tabulate
os.makedirs(LOGS_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(tabulate(df, headers="keys", tablefmt="psql"))

print(f"[Info] Combined logs successfully written to {OUTPUT_FILE}")

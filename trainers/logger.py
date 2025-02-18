#!/usr/bin/env python
import os
import sys
import argparse
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from tabulate import tabulate

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Visualize MLflow Metrics and Logs")
    parser.add_argument("--experiment_name", type=str, required=True, 
                        help="Name of the MLflow experiment to visualize")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    # Get the experiment from MLflow.
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[Error] Experiment '{experiment_name}' not found!")
        sys.exit(1)
    experiment_id = experiment.experiment_id

    # Search for runs under this experiment.
    runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs_df.empty:
        print(f"[Error] No runs found for experiment '{experiment_name}'.")
        sys.exit(1)

    # Create output folder structure: images/<experiment_name>
    output_folder = os.path.join("images", experiment_name)
    os.makedirs(output_folder, exist_ok=True)

    # Format the run dataframe nicely using tabulate.
    logs_table = tabulate(runs_df, headers="keys", tablefmt="psql", showindex=False)
    log_file_path = os.path.join(output_folder, "logs.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(logs_table)
    print(f"[Info] Logs written to {log_file_path}")

    # Initialize MLflow client to get metric histories.
    client = MlflowClient()

    # For simplicity, we use the first run in the experiment.
    run_id = runs_df.iloc[0]["run_id"]
    print(f"[Info] Using run id: {run_id}")

    # Retrieve the run object to access single-value metrics.
    run = client.get_run(run_id)

    # Check for final (single-value) metrics that the training script might have logged.
    total_training_time = run.data.metrics.get("total_training_time_s", None)
    if total_training_time is not None:
        print(f"[Info] Total training time: {total_training_time:.2f} seconds")
    else:
        print("[Info] No 'total_training_time_s' metric found for this run.")

    total_run_time = run.data.metrics.get("total_run_time_s", None)
    if total_run_time is not None:
        print(f"[Info] Total run time: {total_run_time:.2f} seconds")
    else:
        print("[Info] No 'total_run_time_s' metric found for this run.")

    # Define the metrics we want to visualize per step.
    metric_names = [
        "train_loss", 
        "val_loss", 
        "iteration_time_ms", 
        "mfu", 
        "tokens_processed", 
        "tokens_per_second"  # might not exist, but we keep it for completeness
    ]
    
    # Dictionary to hold metric histories.
    metric_history = {}

    # Query metric history for each metric.
    for metric in metric_names:
        history = client.get_metric_history(run_id, metric)
        if not history:
            print(f"[Warning] No history found for metric: {metric}")
            continue
        steps = [m.step for m in history]
        values = [m.value for m in history]

        # Store the steps/values for later combined plots (e.g., train vs val loss).
        metric_history[metric] = (steps, values)

        # Plot the metric.
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, marker='o', linestyle='-')
        plt.title(f"{metric} over Steps")
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        metric_path = os.path.join(output_folder, f"{metric}.png")
        plt.savefig(metric_path)
        plt.close()
        print(f"[Info] Saved graph for {metric} at {metric_path}")

    # If both train and validation loss exist, plot them together.
    if "train_loss" in metric_history and "val_loss" in metric_history:
        steps_train, train_values = metric_history["train_loss"]
        steps_val, val_values = metric_history["val_loss"]
        plt.figure(figsize=(8, 5))
        plt.plot(steps_train, train_values, marker='o', linestyle='-', label="Train Loss")
        plt.plot(steps_val, val_values, marker='o', linestyle='-', label="Validation Loss")
        plt.title("Train vs Validation Loss over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_comp_path = os.path.join(output_folder, "loss_comparison.png")
        plt.savefig(loss_comp_path)
        plt.close()
        print(f"[Info] Saved train/validation loss comparison graph at {loss_comp_path}")

    print(f"[Info] All graphs and logs have been saved in: {output_folder}")

if __name__ == "__main__":
    main()

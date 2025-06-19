import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

def plot_dist_labeled(df, valid_test_results_10, path_to_output):
    valid_df = df[df["VeloOut"] != -1]
    true_speeds = valid_df["VeloOut"]
    predicted_valid_speeds = valid_test_results_10["final prediction"]
    
    # Define bins
    bins = np.linspace(min(true_speeds.min(), predicted_valid_speeds.min()),
                       max(true_speeds.max(), predicted_valid_speeds.max()), 31)
    
    # Get histogram data
    true_counts, _ = np.histogram(true_speeds, bins=bins)
    pred_counts, _ = np.histogram(predicted_valid_speeds, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.4  # bar width
    
    # Define figure and font settings
    plt.figure(figsize=(6.3, 4.0))  # ~16 cm wide
    plt.rcParams.update({'font.size': 10})  # match body text size
    
    # Create grouped bars
    plt.bar(bin_centers - width/2, true_counts, width=width, label="True Speeds", edgecolor='black')
    plt.bar(bin_centers + width/2, pred_counts, width=width, label="Predicted Speeds", edgecolor='black')
    
    # Labeling
    plt.title("True vs. Predicted Valid Bubble Speeds", fontsize=12)
    plt.xlabel("Velocity (m/s)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save
    plot_file_path = os.path.join(path_to_output, 'images/Distribution valid.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_dist_all(df, outcome_df, path_to_output):
    valid_df = df[df["VeloOut"] != -1]
    cert_outcome_df = outcome_df[outcome_df['Standard deviation %'] < 10]
    
    # Match index if necessary
    true_speeds = valid_df["VeloOut"]
    predicted_speeds = cert_outcome_df["final prediction"]
    # Plot
    bins = np.linspace(min(true_speeds.min(), predicted_speeds.min()),
                       max(true_speeds.max(), predicted_speeds.max()), 31)
    
    # Compute histogram counts
    true_counts, _ = np.histogram(true_speeds, bins=bins)
    pred_counts, _ = np.histogram(predicted_speeds, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.4  # bar width
    
    # Set figure size and font
    plt.figure(figsize=(6.3, 4.0))  # ~16 cm wide
    plt.rcParams.update({'font.size': 10})
    
    # Grouped bars
    plt.bar(bin_centers - width/2, true_counts, width=width, label="Original", edgecolor='black')
    plt.bar(bin_centers + width/2, pred_counts, width=width, label="Ensemble", edgecolor='black')
    
    # Labels and layout
    plt.title("Distribution of Bubble Speeds: Original vs Ensemble", fontsize=12)
    plt.xlabel("Velocity (m/s)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plot_file_path = os.path.join(path_to_output, 'images/Distribution all.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_parity_all(y_velo, outcome_df_valid, path_to_output):
    plt.figure(figsize=(7, 7))

    # Define parity line bounds
    min_val = y_velo.min()
    max_val = y_velo.max()
    x_vals = np.linspace(min_val, max_val, 100)
    
    # Perfect parity line
    plt.plot(x_vals, x_vals, 'k--', label='Perfect parity', lw=2)
    
    # ±10% error bounds
    plt.fill_between(x_vals, x_vals * 0.9, x_vals * 1.1, color='gray', alpha=0.2, label='±10% error margin')
    
    # Scatter points
    plt.scatter(y_velo, outcome_df_valid['predictions model 1'], color='blue', alpha=0.2, s=10, label='Model 1')
    plt.scatter(y_velo, outcome_df_valid['predictions model 2'], color='green', alpha=0.2, s=10, label='Model 2')
    plt.scatter(y_velo, outcome_df_valid['predictions model 3'], color='red', alpha=0.2, s=10, label='Model 3')
    plt.scatter(y_velo, outcome_df_valid['predictions model 4'], color='purple', alpha=0.2, s=10, label='Model 4')
    plt.scatter(y_velo, outcome_df_valid['predictions model 5'], color='orange', alpha=0.2, s=10, label='Model 5')
    plt.scatter(y_velo, outcome_df_valid['final prediction'], color='black', alpha=0.7, s=15, label='Unfiltered Ensemble')
    
    # Labels and styling
    plt.xlabel("True Velocity(m/s)", fontsize=12)
    plt.ylabel("Predicted Velocity(m/s)", fontsize=12)
    plt.title("Parity Plot of Model Predictions vs. True Labels", fontsize=14)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True)
    plot_file_path = os.path.join(path_to_output, 'images/Parity_plot_all.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_parity_separate(y_velo, outcome_df_valid, path_to_output):
    predictions = [
    ('Model 1', outcome_df_valid['predictions model 1'], 'blue'),
    ('Model 2', outcome_df_valid['predictions model 2'], 'green'),
    ('Model 3', outcome_df_valid['predictions model 3'], 'red'),
    ('Model 4', outcome_df_valid['predictions model 4'], 'purple'),
    ('Model 5', outcome_df_valid['predictions model 5'], 'orange'),
    ('Unfiltered Ensemble', outcome_df_valid['final prediction'], 'black'),
    ]
    
    # Plot
    n_cols = 3
    n_rows = (len(predictions) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    for idx, (name, pred, color) in enumerate(predictions):
        ax = axes.flat[idx]
        ax.plot([y_velo.min(), y_velo.max()], [y_velo.min(), y_velo.max()], 'k--', lw=1.5)
        ax.scatter(y_velo, pred, color=color, alpha=0.5, s=15)
        ax.set_title(f"Parity Plot: {name}", fontsize=12)
        ax.set_xlabel("True Velocity(m/s)", fontsize=10)
        ax.set_ylabel("Predicted Velocity(m/s)", fontsize=10)
        ax.grid(True)
    
    # Hide unused subplots
    for idx in range(len(predictions), n_rows * n_cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plot_file_path = os.path.join(path_to_output, 'images/Parity_plot_separate.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_parity_ensemble(y_velo, outcome_df_valid, path_to_output):
    plt.figure(figsize=(7, 7))

    # Define parity line bounds
    min_val = y_velo.min()
    max_val = y_velo.max()
    x_vals = np.linspace(min_val, max_val, 100)
    
    # Perfect parity line
    plt.plot(x_vals, x_vals, 'k--', label='Perfect parity', lw=2)
    
    merged_df_valid = outcome_df_valid
    merged_df_valid['True'] = y_velo
    accepted_df_valid = merged_df_valid[merged_df_valid['Standard deviation %'] < 10]
    # Scatter points
    plt.scatter(accepted_df_valid['True'], accepted_df_valid['final prediction'], color='black', alpha=0.5, s=15, label='Ensemble')
    
    # Labels and styling
    plt.xlabel("True Velocity(m/s)", fontsize=12)
    plt.ylabel("Predicted Velocity(m/s)", fontsize=12)
    plt.title("Accepted Ensemble", fontsize=14)
    plt.grid(True)
    plot_file_path = os.path.join(path_to_output, 'images/Parity_plot_ensemble.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_check_accepted(outcome_df, df_check, df, path_to_output):
    outcome_df_voltage = outcome_df.copy()
    outcome_df_voltage['signal'] = df_check["VoltageOut"]
    outcome_df_voltage['true'] = df['VeloOut']
    
    # Filter for unlabeled bubbles with acceptable uncertainty
    unlabeled_df = outcome_df_voltage[outcome_df_voltage['true'] == -1]
    unlabeled_df_accepted = unlabeled_df[unlabeled_df['Standard deviation %'] < 10].reset_index(drop=True)
    
    # Randomly select samples from the accepted set
    n_samples = 6
    random_seed = 1
    np.random.seed(random_seed)
    random_indices = np.random.choice(len(unlabeled_df_accepted), size=min(n_samples, len(unlabeled_df_accepted)), replace=False)
    
    # Plotting
    n_cols = 3
    n_rows = (len(random_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    
    letters = ['A','B','C','D','E','F']
    for idx, random_idx in enumerate(random_indices):
        ax = axes.flat[idx]
        sample = unlabeled_df_accepted.iloc[random_idx]
        voltage_data = sample['signal']
        predicted_speed = sample['final prediction']
        std_pct = sample['Standard deviation %']
    
        if isinstance(voltage_data, list):
            ax.plot(voltage_data, color='green', alpha=0.7)
            ax.set_title(f'{letters[idx]}, std% = {std_pct:.2f}', fontsize=12)
    
            if idx % n_cols == 0:
                ax.set_ylabel("Voltage (V)", fontsize=10)
            else:
                ax.tick_params(labelleft=False)
    
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=10)
            else:
                ax.tick_params(labelbottom=False)
        else:
            ax.text(0.5, 0.5, f"Invalid data for Bubble {random_idx}", ha='center', va='center', fontsize=6)
            ax.set_axis_off()
    
    # Remove empty subplots
    for idx in range(len(random_indices), n_rows * n_cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plot_file_path = os.path.join(path_to_output, 'images/Check_plot_accepted.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()

def plot_check_rejected(outcome_df, df_check, df, path_to_output):
    outcome_df_voltage = outcome_df.copy()
    outcome_df_voltage['signal'] = df_check["VoltageOut"]
    outcome_df_voltage['true'] = df['VeloOut']
    
    # Filter for unlabeled bubbles with acceptable uncertainty
    unlabeled_df = outcome_df_voltage[outcome_df_voltage['true'] == -1]
    unlabeled_df_accepted = unlabeled_df[unlabeled_df['Standard deviation %'] > 10].reset_index(drop=True)
    
    # Randomly select samples from the accepted set
    n_samples = 6
    random_seed = 18
    np.random.seed(random_seed)
    random_indices = np.random.choice(len(unlabeled_df_accepted), size=min(n_samples, len(unlabeled_df_accepted)), replace=False)
    
    # Plotting
    n_cols = 3
    n_rows = (len(random_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    
    letters = ['G','H','I','J','K','L']
    for idx, random_idx in enumerate(random_indices):
        ax = axes.flat[idx]
        sample = unlabeled_df_accepted.iloc[random_idx]
        voltage_data = sample['signal']
        predicted_speed = sample['final prediction']
        std_pct = sample['Standard deviation %']
    
        if isinstance(voltage_data, list):
            ax.plot(voltage_data, color='red', alpha=0.7)
            ax.set_title(f'{letters[idx]}, std% = {std_pct:.2f}', fontsize=12)
    
            if idx % n_cols == 0:
                ax.set_ylabel("Voltage (V)", fontsize=10)
            else:
                ax.tick_params(labelleft=False)
    
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=10)
            else:
                ax.tick_params(labelbottom=False)
        else:
            ax.text(0.5, 0.5, f"Invalid data for Bubble {random_idx}", ha='center', va='center', fontsize=6)
            ax.set_axis_off()
    
    # Remove empty subplots
    for idx in range(len(random_indices), n_rows * n_cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plot_file_path = os.path.join(path_to_output, 'images/Check_plot_rejected.png')
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")
    plt.show()
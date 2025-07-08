# Libary imports
import torch
import numpy as np
import os

# Function imports
from advanced_dataloading import (
    process_folder,
    process_folder_check
)
from models import (
    load_scalers,
    load_models,
    predict
)
from visualization import (
    plot_dist_labeled,
    plot_dist_all,
    plot_parity_all,
    plot_parity_separate,
    plot_parity_ensemble,
    plot_check_accepted,
    plot_check_rejected
)


""" Start user input """
path_to_data = r'U:\Bubble Column\Data\2411_Xray alcohols\Fiber Probe\241108 - Water center of column'
path_to_output = os.path.join(os.path.expanduser("~"), r"Codebase\NN-FiberProbe\Output")
files = [3]
dist_plots = True
parity_plots = True
check_plots = True
""" End user input """


# Load the models and scalers
gru1, gru2, lstm, cnn1, cnn2 = load_models()
feature_scaler, target_scaler = load_scalers()

# Load and prepare the data
df = process_folder(path_to_data, path_to_output, files=files, plot=False, labels=True,)
X_array = np.vstack(df["VoltageOut"].to_numpy())  # shape: (samples, timesteps)
X_scaled = feature_scaler.transform(X_array)      # apply your trained scaler
X_tensor = torch.tensor(X_scaled[..., np.newaxis], dtype=torch.float32)  # shape: (samples, timesteps, 1)

# Make Predictions
outcome_df, outcome_df_valid, valid_test_results_10, y_velo = predict(
    gru1,
    gru2,
    lstm,
    cnn1,
    cnn2,
    target_scaler,
    feature_scaler,
    X_tensor,
    df
)

# Plotting
if dist_plots:
    plot_dist_labeled(df, valid_test_results_10, path_to_output)
    plot_dist_all(df, outcome_df, path_to_output)

if parity_plots:
    plot_parity_all(y_velo, outcome_df_valid, path_to_output)
    plot_parity_separate(y_velo, outcome_df_valid, path_to_output)
    plot_parity_ensemble(y_velo, outcome_df_valid, path_to_output)

if check_plots:
    df_check = process_folder_check(path_to_data, path_to_output, files=files, plot=False, labels=True,)
    plot_check_accepted(outcome_df, df_check, df, path_to_output)
    plot_check_rejected(outcome_df, df_check, df, path_to_output)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks

########################################################
# TABLE OF CONTENTS

# find_all_runs: Finds all runs in the folder path.
# get_binlogdata: Parse `.binlog` files to extract metadata such as coefficients, frequency, and comments.
# get_labels: Extract bubble labels from `.evt` files with valid velocities (`VeloOut != -1`).
# get_bubbles_advanced: Extract bubble entries and exits using a dual-thresholding strategy.
#                       Includes downsampling, smoothing, gradient computation, and peak detection.
#                       Optionally plots results and saves the plot.
# get_bubbles_advanced_check: Same as get_bubbles_advanced, but selects a bigger ROI, for visualization purposes
# plot_bubble_detection: Visualize voltage data, detected peaks, and entry/exit points.
#                        Saves the plot in the specified folder.
# save_bubbles: Save extracted bubble data into a CSV file. Match bubble data with labels and identify missing labels.
# process_folder: Process a single folder containing bubble run data and generate a CSV.
# process_folder_check: same as process_folder but uses get_bubbles_advanced_check.
########################################################


def find_all_runs(folder_path):
    """Find all sets of (.bin, .binlog, .evt) files in the given folder.

    Args:
        folder_path (str): Path to the folder containing run files.

    Returns:
        tuple[list[str] | None, list[str] | None, list[str] | None, list[str]]:
            bin_files: List of .bin file paths.
            binlog_files: List of .binlog file paths.
            evt_files: List of .evt file paths.
            run_names: List of run names.
    """
    runs = {}

    for file in os.listdir(folder_path):
        if file.endswith(".bin") and "_stream" not in file:
            run_name = os.path.splitext(file)[0]
            if run_name not in runs:
                runs[run_name] = {"bin": None, "binlog": None, "evt": None}
            runs[run_name]["bin"] = os.path.join(folder_path, file)

        elif file.endswith(".binlog"):
            name = os.path.splitext(file)[0]
            for run in runs:
                if name in run:
                    runs[run]["binlog"] = os.path.join(folder_path, file)

        elif file.endswith(".evt") and "_stream" not in file:
            name = os.path.splitext(file)[0]
            for run in runs:
                if name in run:
                    runs[run]["evt"] = os.path.join(folder_path, file)

    # Build output lists
    bin_files = []
    binlog_files = []
    evt_files = []
    run_names = []

    for run_name, files in runs.items():
        bin_files.append(files["bin"])
        binlog_files.append(files["binlog"])
        evt_files.append(files["evt"])
        run_names.append(run_name)

    return bin_files, binlog_files, evt_files, run_names


def get_binlogdata(binlog_file):
    """Extracts binlogdata from .binlog.

    Args:
        binlog_file (str): Path to binlog file (.binlog).

    Returns:
        dict: Metadata including channelCoef1, channelCoef2, acquisitionFrequency, flowRate, and bin_file.
    """
    tree = ET.parse(binlog_file)
    root = tree.getroot()

    acquisition_comment = root.attrib.get('acquisitionComment', '')
    flow_rate_match = re.search(r'(\d+)\s*[lL][/-]?[mM]in', acquisition_comment)
    if flow_rate_match:
        flow_rate = int(flow_rate_match.group(1))
    else:
        flow_rate = -1

    binlogdata = {
        "channelCoef1": float(root.find(".//channel").attrib['channelCoef1']),
        "channelCoef2": float(root.find(".//channel").attrib['channelCoef2']),
        "acquisitionFrequency": float(root.attrib['acquisitionFrequency']),
        "flowRate": flow_rate,
        "bin_file": root.find(".//channel").attrib['channelOutputFile']
    }

    print("Binlog data extracted")
    return binlogdata


def get_labels(evt_file):
    """Extracts bubble labels with VeloOut != -1 from the evt_file.

    Args:
        evt_file (str): Path to eventlog file (.evt).

    Returns:
        list[list[str | int | float]]: List of labels as [L_idx, Exit, VeloOut] where VeloOut != -1.
    """
    with open(evt_file, 'rb') as file:
        content = file.read()

    lines = content.decode('latin1').splitlines()
    data = [line.split('\t') for line in lines]

    headers, rows = data[0], data[1:]
    exit_idx = headers.index("Exit")
    veloout_idx = headers.index("VeloOut")

    extracted_bubbles = []
    valid_idx = 0

    for row in rows:
        # Extract and process Exit and VeloOut fields
        exit_value = int(row[exit_idx])
        veloout_value = float(row[veloout_idx].replace(",", "."))

        # Include only labels where VeloOut != -1
        if veloout_value != -1:
            extracted_bubbles.append(["L" + str(valid_idx), exit_value, veloout_value])
            valid_idx += 1

    print(f"LABELS: {len(extracted_bubbles)} bubble labels with VeloOut != -1 extracted.")
    return extracted_bubbles


def get_bubbles_advanced(bin_file, coef1, coef2, plot=False, folder_path=None,
                         run_name=None):
    """Extracts bubble entries and exits implementing dual-thresholding strategy.

    Args:
        bin_file (str): Path to the binary file (.bin).
        coef1 (float): Channel coefficient 1 (offset).
        coef2 (float): Channel coefficient 2 (scaling factor).
        plot (bool, optional): Whether to plot the results. Defaults to False.
        folder_path (str | None, optional): Path to the folder where the plot
            will be saved. Required if plot is True. Defaults to None.
        run_name (str | None, optional): Name of the run for naming the plot
            file. Required if plot is True. Defaults to None.

    Returns:
        list[list[str | int | float | list[float]]]: Extracted bubble data.
    """
    trans_data = np.memmap(bin_file, dtype=">i2", mode="r")
    voltage_data = (trans_data.astype(np.float32) * coef2 + coef1)
    print(f"{len(voltage_data)} datapoints extracted")

    downsample_factor = 5
    voltage_data_downsampled = voltage_data[::downsample_factor]

    # Apply moving average for additional smoothing
    window_size = 100
    kernel = np.ones(window_size) / window_size
    smoothed_voltage_data = np.convolve(voltage_data_downsampled, kernel, mode='valid')
    smoothed_voltage_data = np.concatenate((np.full(window_size - 1, smoothed_voltage_data[0]), smoothed_voltage_data))

    # Compute the gradient of the smoothed and averaged data
    gradient = np.gradient(smoothed_voltage_data)

    # Detect peaks in the negative gradient
    peaks, _ = find_peaks(-gradient, prominence=0.005, distance=1000)

    tE = peaks * downsample_factor
    tE1 = tE - 500
    tE1 = tE1[tE1 >= 0]

    tE0 = tE1 - 600
    tE0 = tE0[tE0 >= 0]

    bubbles = []
    for idx, (start, end, peak) in enumerate(zip(tE0, tE1, tE)):
        if start >= 0 and end < len(voltage_data):
            voltage_out = voltage_data[start:end].tolist()
            bubbles.append(["E"+str(idx), peak, voltage_out])

    # Plot if requested
    if plot:
        if folder_path is None or run_name is None:
            raise ValueError("Both `folder_path` and `run_name` must be provided when plot=True.")
        plot_bubble_detection(voltage_data, tE, tE1, tE0, n=5000000,
                              folder_path=folder_path, run_name=run_name)

    return bubbles


def get_bubbles_advanced_check(bin_file, coef1, coef2, plot=False,
                               folder_path=None, run_name=None):
    """Extracts bubble entries and exits implementing dual-thresholding strategy (for visualization).

    Args:
        bin_file (str): Path to the binary file (.bin).
        coef1 (float): Channel coefficient 1 (offset).
        coef2 (float): Channel coefficient 2 (scaling factor).
        plot (bool, optional): Whether to plot the results. Defaults to False.
        folder_path (str | None, optional): Path to the folder where the plot
            will be saved. Required if plot is True. Defaults to None.
        run_name (str | None, optional): Name of the run for naming the plot
            file. Required if plot is True. Defaults to None.

    Returns:
        list[list[str | int | float | list[float]]]: Extracted bubble data.
    """
    trans_data = np.memmap(bin_file, dtype=">i2", mode="r")
    voltage_data = (trans_data.astype(np.float32) * coef2 + coef1)
    print(f"{len(voltage_data)} datapoints extracted")

    downsample_factor = 5
    voltage_data_downsampled = voltage_data[::downsample_factor]

    # Apply moving average for additional smoothing
    window_size = 100
    kernel = np.ones(window_size) / window_size
    smoothed_voltage_data = np.convolve(voltage_data_downsampled, kernel, mode='valid')
    smoothed_voltage_data = np.concatenate((np.full(window_size - 1, smoothed_voltage_data[0]), smoothed_voltage_data))

    # Compute the gradient of the smoothed and averaged data
    gradient = np.gradient(smoothed_voltage_data)

    # Detect peaks in the negative gradient
    peaks, _ = find_peaks(-gradient, prominence=0.005, distance=1000)

    tE = peaks * downsample_factor
    tE1 = tE
    tE1 = tE1[tE1 >= 0]

    tE0 = tE1 - 1600
    tE0 = tE0[tE0 >= 0]

    bubbles = []
    for idx, (start, end, peak) in enumerate(zip(tE0, tE1, tE)):
        if start >= 0 and end < len(voltage_data):
            voltage_out = voltage_data[start:end].tolist()
            bubbles.append(["E"+str(idx), peak, voltage_out])

    # Plot if requested
    if plot:
        if folder_path is None or run_name is None:
            raise ValueError("Both `folder_path` and `run_name` must be provided when plot=True.")
        plot_bubble_detection(voltage_data, tE, tE1, tE0, n=5000000, folder_path=folder_path, run_name=run_name)

    return bubbles


def plot_bubble_detection(voltage_data, tE, tE1, tE0, n, folder_path, run_name):
    """Plots and saves the results of voltage data and detected peaks

    Closes the plot after to free up memory and allow the program to continue
    Currently uses standard matplotlib backend, might consider changing if plots
    don't need to be shown. Standard backend tends to leak memory in such a case.

    Args:
        voltage_data (np.ndarray): Original voltage data.
        tE (np.ndarray): Detected peaks in original indices.
        tE1 (np.ndarray): Entry indices.
        tE0 (np.ndarray): Exit indices.
        n (int): Number of points to plot from the original voltage data.
        folder_path (str): Path to the folder where the plot should be saved.
        run_name (str): Name of the current run to use for naming the plot file.
    """
    # Create the plot
    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(len(voltage_data[:n])), voltage_data[:n], label="Original Voltage Data", color="blue", alpha=0.3)

    # Plot tE, tE1, and tE0 within the first `n` points
    valid_tE = tE[tE < n]
    plt.scatter(valid_tE, voltage_data[valid_tE], color="red", label="Detected Peaks (tE)", marker="x", s=50)

    valid_tE1 = tE1[tE1 < n]
    plt.scatter(valid_tE1, voltage_data[valid_tE1], color="purple", label="Exit (tE1)", marker="o", s=50)

    valid_tE0 = tE0[tE0 < n]
    plt.scatter(valid_tE0, voltage_data[valid_tE0], color="pink", label="Entry (tE0)", marker="o", s=50)

    # Labels and title
    plt.title("Voltage Data with Detected Peaks and Shifts")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage")
    plt.legend()

    # Save the plot to the folder
    plot_file_name = f"{run_name}_bubbles_plot.png"
    plot_file_path = os.path.join(folder_path, plot_file_name)
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")

    # Close the plot to free memory and allow the code to continue
    plt.close()


# TODO consider splitting save_bubbles into one version with labels and one without
# Logic is a bit hard to follow with all the nested if statements.
def save_bubbles(extracted_bubbles, run_name, folder_path, bubble_labels,
                 flow_rate, frequency):
    """Saves extracted bubble data to a Pandas DataFrame and identifies missing labels.

    Args:
        extracted_bubbles (list): A list of bubbles, where each bubble is [Bidx, tE, VoltageOut].
        run_name (str | None): Name of the run for file naming.
        folder_path (str): Path to the folder where the CSV will be saved.
        bubble_labels (list | None): List of labels where each label is [Lidx, ExitIdx, VeloOut].
        flow_rate (int): Flow rate of measurement in L/min.
        frequency (float): Frequency of the measurement.

    Returns:
        pd.DataFrame: A DataFrame containing [bubble_idx, B_idx, L_idx, VeloOut, VoltageOut, flowRate, Frequency].
    """
    rows = []
    if bubble_labels:
        missing_labels = []

    # Iterate through each bubble in the extracted bubbles
    for bubble_idx, (E_idx, tE, VoltageOut) in enumerate(extracted_bubbles):
        # bubble_labels can be a list with entries, empty list or None. Latter two go to `else`
        if bubble_labels:
            # Check if any label's ExitIdx is within the bubble's range
            matched_label = None
            for label in bubble_labels:
                L_idx, Exit_idx, VeloOut = label
                if tE - 1000 <= Exit_idx <= tE + 1000:
                    matched_label = (L_idx, VeloOut)
                    break

            # If a matching label is found, use its values
            if matched_label:
                L_idx, VeloOut = matched_label
            else:
                L_idx, VeloOut = -1, -1
        else:
            # No labels provided
            L_idx, VeloOut = -1, -1

        # Append the bubble information to the rows
        rows.append({
            "bubble_idx": str(bubble_idx)+"_"+run_name,
            "E_idx": E_idx,
            "L_idx": L_idx,
            "VeloOut": VeloOut,
            "VoltageOut": VoltageOut,
            "FlowRate": flow_rate,
            "Frequency": frequency
        })

    # Identify missing labels
    if bubble_labels:
        for label in bubble_labels:
            L_idx, Exit_idx, VeloOut = label
            found = False
            for _, tE, _ in extracted_bubbles:
                if tE - 1000 <= Exit_idx <= tE + 1000:
                    found = True
                    break
            if not found:
                missing_labels.append(label)

    # Create a DataFrame
    saved_bubbles = pd.DataFrame(rows)

    # Save the DataFrame to a file in the specified folder
    if run_name:
        file_name = os.path.join(folder_path, f"{flow_rate}_{run_name}_bubbles.csv")
    else:
        file_name = os.path.join(folder_path, f"{flow_rate}_bubbles.csv")

    saved_bubbles.to_csv(file_name, index=False, sep=";")
    print(f"Saved bubbles to {file_name}")

    if bubble_labels:
        # Print missing labels
        if missing_labels:
            print("\nMissing Labels:")
            for label in missing_labels:
                print(f"L_idx: {label[0]}, ExitIdx: {label[1]}, VeloOut: {label[2]}")
        else:
            print("No missing labels.")

    # Count and print bubbles with VeloOut != -1
    valid_bubbles = saved_bubbles[saved_bubbles["VeloOut"] != -1]
    print(f"EXTRACTED: {len(valid_bubbles)} bubbles have VeloOut != -1 out of {len(saved_bubbles)} total bubbles.")
    print(saved_bubbles.head())

    return saved_bubbles


def process_folder(input_path, output_path, files, plot, labels):
    """Processes selected runs in a folder and returns a combined DataFrame.

    Args:
        input_path (str): Path to the folder containing the runs.
        output_path (str): Path to the folder where you want the results to be saved.
        files (list[int]): List of the runs you want to be used from the folder, starting with 0.
        plot (bool): If True, plots the results of the voltage data and all
            detected peaks, saves the plot, and allows code execution to continue.
        labels (bool): If True, extracts bubble labels with VeloOut != -1 from the evt_file.

    Returns:
        pd.DataFrame: A DataFrame containing [bubble_idx, B_idx, L_idx, VeloOut, VoltageOut, flowRate, Frequency].
    """
    bin_files, binlog_files, evt_files, run_names = find_all_runs(input_path)
    all_dfs = []

    for i in files:
        binlogdata = get_binlogdata(binlog_files[i])
        coef1 = binlogdata["channelCoef1"]
        coef2 = binlogdata["channelCoef2"]
        flowRate = binlogdata["flowRate"]
        acquisitionFrequency = binlogdata["acquisitionFrequency"]

        print(f"Processing run: {run_names[i]}")
        print(binlogdata)

        extracted_bubbles = get_bubbles_advanced(
            bin_files[i], coef1, coef2, plot, output_path, run_names[i]
        )

        bubble_labels = get_labels(evt_files[i]) if labels else None

        df = save_bubbles(
            extracted_bubbles, run_names[i], output_path,
            bubble_labels, flowRate, acquisitionFrequency
        )
        all_dfs.append(df)

    # Combine all DataFrames
    final_df = pd.concat(all_dfs, ignore_index=True)

    return final_df


def process_folder_check(input_path, output_path, files, plot, labels):
    """Processes selected runs in a folder and returns a combined DataFrame (for visualization).

    Args:
        input_path (str): Path to the folder containing the runs.
        output_path (str): Path to the folder where you want the results to be saved.
        files (list[int]): List of the runs you want to be used from the folder, starting with 0.
        plot (bool): If True, plots the results of the voltage data and all
            detected peaks, saves the plot, and allows code execution to continue.
        labels (bool): If True, extracts bubble labels with VeloOut != -1 from the evt_file.

    Returns:
        pd.DataFrame: A DataFrame containing [bubble_idx, B_idx, L_idx, VeloOut,
            VoltageOut, flowRate, Frequency].
    """
    bin_files, binlog_files, evt_files, run_names = find_all_runs(input_path)
    all_dfs = []

    for i in files:
        binlogdata = get_binlogdata(binlog_files[i])
        coef1 = binlogdata["channelCoef1"]
        coef2 = binlogdata["channelCoef2"]
        flowRate = binlogdata["flowRate"]
        acquisitionFrequency = binlogdata["acquisitionFrequency"]

        print(f"Processing run: {run_names[i]}")
        print(binlogdata)

        extracted_bubbles = get_bubbles_advanced_check(
            bin_files[i], coef1, coef2, plot, output_path, run_names[i]
        )

        bubble_labels = get_labels(evt_files[i]) if labels else None

        df = save_bubbles(
            extracted_bubbles, run_names[i], output_path,
            bubble_labels, flowRate, acquisitionFrequency
        )
        all_dfs.append(df)

    # Combine all DataFrames
    final_df = pd.concat(all_dfs, ignore_index=True)

    return final_df

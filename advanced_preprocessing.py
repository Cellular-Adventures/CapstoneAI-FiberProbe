from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
from sklearn.model_selection import KFold

import os
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import ast
import pickle


########################################################
# TABLE OF CONTENTS

# valid_velo_data: returns data and labels with only valid velocities
# random_noise: duplicates and randomly adds noise to some data
# bin_data: bins all y labels as data (does not regard X data) > part of flatten_data_distribution
# calculate_duplication_factors: calculates how to scale data in bins > part of flatten_data_distribution
# duplicate_and_augment_data: duplicates and augments data based on bin frequency > part of flatten_data_distribution
# flatten_data_distribution: flattens the data distribution according to bin sizes, by augmenting and duplicating data
# split_scale_save: splits the data into TVT sets, scales the data and provides the scalers
########################################################



def valid_velo_data(data):
    """
    Extracts the data with valid velocities. Only works for pandas DataFrames right now.

    Args:
        data: DataFrame with bubble voltages and labels.

    Returns:
        x: Numpy array with voltage data of all valid bubbles.
        y: Numpy array with labels of all valid bubbles.
    """
    valid = data[data["VeloOut"] != -1]
    x = valid["VoltageOut"].tolist()
    y = valid["VeloOut"].astype(float).tolist()
    return np.array(x), np.array(y)



def random_noise(data, labels, chance, noise_level=0.005, random_seed=None):
    """
    Duplicates some samples and adds noise to them.
    
    Args:
        data: 2D Numpy array (or list) with features of the samples
        labels: 1D Numpy array (or list) with labels per sample
        chance: fraction of data (between 0 and 1) that will get augmented and duplicated
        noise_level: standard deviation in the noise. Must be non-negative. 

    Output:
        x: Numpy array with the duplicated/transformed samples appended
        y: Numpy array with the duplicated labels appended
    """

    if not (0. <= chance):
        raise ValueError("Chance should be larger than 0")
    if len(data) != len(labels):
        raise ValueError("data and labels should be the same length")

    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)

    # set random seed if applicable
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # amount of images to be augmented:
    n = int(chance * len(data))

    # creating an array of length n with random numbers 
    random_list = np.random.randint(0, len(data), n)

    x_new = data
    y_new = labels
    for rand in random_list:
        duplicate = data[rand] + np.random.normal(0, noise_level, data[rand].shape)
        duplicate_label = labels[rand]
        x_new = np.concatenate([x_new, [duplicate]])
        y_new = np.append(y_new, duplicate_label)

    return x_new, y_new


def bin_data(y, bins):
    """
    Makes bins based on y data (velocities)

    Args: 
        y: numpy array with labels
        bins: number of bins

    Output:
        hist: array that contains counts of datapoints in each bin.
        bin_indices: array that indicates which bin each data point in y belongs to.
    
    """
    hist, bin_edges = np.histogram(y, bins=bins)
    bin_indices = np.digitize(y, bin_edges[:-1])
    return hist, bin_indices


def calculate_duplication_factors(hist, scale_factor=0.5):
    """
    Calculates the factors that each bin should be duplicated with. Less frequent bins will get duplicated more.

    Args:
        hist: array with samples per bin
        scale_factor: determines how much the distribution will be flattened. 
                        1=almost completely flat, 0=no flattening

    Output:
        array with the factors per bin
        
    """
    max_freq = np.max(hist)
    factors = np.zeros_like(hist, dtype=float)
    # preventing division by 0
    non_zero_indices = hist > 0

    # scaling factor so less frequent data does not get fully duplicated 10 times.
    factors[non_zero_indices] = (max_freq / hist[non_zero_indices]) * scale_factor
    # No duplication for the most frequent bin
    factors[hist == max_freq] = 1 

    return factors


def duplicate_and_augment_data(X, y, bin_indices, factors, noise=0.005):
    """
    Duplicates/augments the data per bin, scaled with the size of each bin 
    (smaller bin -> more duplication)

    Args:
        X: X data
        y: y data
        bin_indices: array with which datapoints correspond to which bins (from bin_data)

    Output:
        augmented_X: lengthened and partly augmented X data
        augmented_y: lenghthened y data of the augmented_X data
    
    """
    augmented_X = X.copy()
    augmented_y = y.copy()
    for i, (x_value, y_value) in enumerate(zip(X, y)):
        bin_idx = bin_indices[i] - 1
        factor = factors[bin_idx]
        for _ in range(int(factor) - 1):
            if np.random.rand() < 0.5:
                x_new, y_new = random_noise([x_value], [y_value], chance=1, noise_level=noise)
            else:
                x_new, y_new = random_flip([x_value], [y_value], chance=1)
            augmented_X = np.concatenate([augmented_X, x_new])
            augmented_y = np.concatenate([augmented_y, y_new])
    return augmented_X, augmented_y


def flatten_data_distribution(X, y, bins, scaling_factor=0.5, noise=0.005):
    """
    Combines the functions to flatten the distribution (by augmenting data).
    
    Args:
        X: X data
        y: y data
        bins: amount of bins
        scaling_factor: factor that prevents data from becoming all bins becoming the most frequent

    Output:
        augmented_X: lengthened and partly augmented X data
        augmented_y: lenghthened y data of the augmented_X data
    """
    hist, bin_indices = bin_data(y, bins)
    factors = calculate_duplication_factors(hist, scale_factor=scaling_factor)
    augmented_X, augmented_y = duplicate_and_augment_data(X, y, bin_indices, factors, noise=noise)
    return augmented_X, augmented_y



def split_scale_save(data):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X, y = valid_velo_data(data)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.90, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.75, random_state=0)
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    return feature_scaler, target_scaler, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled




    

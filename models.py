import pickle
import numpy as np
import pandas as pd
import torch
from advanced_preprocessing import valid_velo_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUModel(torch.nn.Module):
    """Gated Recurrent Unit (GRU) based regression model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in each GRU layer.
        num_layers (int, optional): Number of stacked GRU layers. Defaults to 2.
    """

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, timesteps, features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1].
        """
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(torch.nn.Module):
    """Long Short-Term Memory (LSTM) based regression model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
    """

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, timesteps, features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1].
        """
        h_0 = torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class CNNModel(torch.nn.Module):
    """1D Convolutional Neural Network (CNN) for regression tasks.

    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        hidden_units (int, optional): Number of filters in each Conv1D layer. Defaults to 32.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        num_layers (int, optional): Number of convolutional layers. Defaults to 2.
        input_length (int, optional): Length of the input sequence. Defaults to 150.
        output_size (int, optional): Number of output units. Defaults to 1.
    """

    def __init__(self, input_channels=1, hidden_units=32, kernel_size=3,
                 num_layers=2, input_length=150, output_size=1):
        super(CNNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        current_channels = input_channels

        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Conv1d(
                    current_channels,
                    hidden_units,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.MaxPool1d(kernel_size=2))
            current_channels = hidden_units

        self._dummy_input = torch.zeros(1, input_channels, input_length)
        with torch.no_grad():
            dummy_output = self._forward_features(self._dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = torch.nn.Linear(flattened_size, 64)
        self.fc2 = torch.nn.Linear(64, output_size)

    def _forward_features(self, x):
        """Passes input through convolutional layers.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, length].

        Returns:
            torch.Tensor: Output tensor after convolutional layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        """Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, timesteps, features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size].
        """
        x = x.permute(0, 2, 1)  # [batch_size, channels, length]
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_scalers():
    """Loads feature and target scalers from disk.

    Returns:
        tuple: (feature_scaler, target_scaler)
    """
    with open('advanced_scalers/feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)

    with open('advanced_scalers/target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    return feature_scaler, target_scaler


def load_models():
    """Loads pre-trained GRU, LSTM, and CNN models from disk.

    Returns:
        tuple: (gru1, gru2, lstm, cnn1, cnn2)
            gru1 (GRUModel): First GRU model.
            gru2 (GRUModel): Second GRU model.
            lstm (LSTMModel): LSTM model.
            cnn1 (CNNModel): First CNN model.
            cnn2 (CNNModel): Second CNN model.
    """
    gru1 = GRUModel(input_size=1, hidden_size=20, num_layers=2)
    gru1.load_state_dict(torch.load("advanced_model_files/gru1.h5", map_location='cpu', weights_only=True))
    gru1.eval()

    gru2 = GRUModel(input_size=1, hidden_size=10, num_layers=3)
    gru2.load_state_dict(torch.load("advanced_model_files/gru2.h5", map_location='cpu', weights_only=True))
    gru2.eval()

    lstm = LSTMModel(input_size=1, hidden_size=18, num_layers=2)
    lstm.load_state_dict(torch.load("advanced_model_files/lstm.h5", map_location='cpu', weights_only=True))
    lstm.eval()

    cnn1 = CNNModel(input_channels=1, hidden_units=32, kernel_size=9, num_layers=7, input_length=600, output_size=1)
    cnn1.load_state_dict(torch.load("advanced_model_files/cnn1.h5", map_location='cpu', weights_only=True))
    cnn1.eval()

    cnn2 = CNNModel(input_channels=1, hidden_units=64, kernel_size=15, num_layers=5, input_length=600, output_size=1)
    cnn2.load_state_dict(torch.load("advanced_model_files/cnn2.h5", map_location='cpu', weights_only=True))
    cnn2.eval()

    return gru1, gru2, lstm, cnn1, cnn2


def predict(gru1, gru2, lstm, cnn1, cnn2, target_scaler, feature_scaler, X_tensor, df):
    """Runs ensemble prediction and evaluation using multiple models.

    Args:
        gru1 (GRUModel): First GRU model.
        gru2 (GRUModel): Second GRU model.
        lstm (LSTMModel): LSTM model.
        cnn1 (CNNModel): First CNN model.
        cnn2 (CNNModel): Second CNN model.
        target_scaler (sklearn.preprocessing object): Scaler for target variable.
        feature_scaler (sklearn.preprocessing object): Scaler for input features.
        X_tensor (torch.Tensor): Input data tensor.
        df (pd.DataFrame): Original dataframe for evaluation.

    Returns:
        tuple:
            outcome_df (pd.DataFrame): Predictions and statistics for all data.
            outcome_df_valid (pd.DataFrame): Predictions and statistics for valid data.
            valid_test_results_10 (pd.DataFrame): Subset with <10% error and <10% std.
            y_velo (np.ndarray): Ground truth target values for valid data.
    """
    with torch.no_grad():
        y_gru1_scaled = gru1(X_tensor)
        y_gru2_scaled = gru2(X_tensor)
        y_lstm_scaled = lstm(X_tensor)
        y_cnn1_scaled = cnn1(X_tensor)
        y_cnn2_scaled = cnn2(X_tensor)
    y_gru1 = target_scaler.inverse_transform(y_gru1_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_gru2 = target_scaler.inverse_transform(y_gru2_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_lstm = target_scaler.inverse_transform(y_lstm_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_cnn1 = target_scaler.inverse_transform(y_cnn1_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_cnn2 = target_scaler.inverse_transform(y_cnn2_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()

    y_pred = ((y_lstm + y_gru1 + y_gru2 + y_cnn1 + y_cnn2) / 5).flatten()
    outcome_df = pd.DataFrame({
        "predictions model 1": y_gru1,
        "predictions model 2": y_gru2,
        "predictions model 3": y_lstm,
        "predictions model 4": y_cnn1,
        "predictions model 5": y_cnn2,
        "final prediction": y_pred})
    outcome_df['Standard deviation'] = outcome_df[[
        "predictions model 1",
        "predictions model 2",
        "predictions model 3",
        "predictions model 4"]].std(axis=1)
    outcome_df['Standard deviation %'] = outcome_df['Standard deviation'] / outcome_df['final prediction'] * 100

    # Evaluation metrics
    valid_bubbles_ai = len(outcome_df[outcome_df['Standard deviation %'] < 10])/len(outcome_df) * 100
    valid_bubbles_boring_software = len(valid_velo_data(df)[0])/len(df) * 100

    X_velo, y_velo = valid_velo_data(df)
    X_velo_scaled = torch.tensor(feature_scaler.transform(X_velo)[..., np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        y_gru1_scaled_velo = gru1(X_velo_scaled)
        y_gru2_scaled_velo = gru2(X_velo_scaled)
        y_lstm_scaled_velo = lstm(X_velo_scaled)
        y_cnn1_scaled_velo = cnn1(X_velo_scaled)
        y_cnn2_scaled_velo = cnn2(X_velo_scaled)
    y_gru1_velo = target_scaler.inverse_transform(y_gru1_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_gru2_velo = target_scaler.inverse_transform(y_gru2_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_lstm_velo = target_scaler.inverse_transform(y_lstm_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_cnn1_velo = target_scaler.inverse_transform(y_cnn1_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_cnn2_velo = target_scaler.inverse_transform(y_cnn2_scaled_velo.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_velo = ((y_lstm_velo + y_gru1_velo + y_gru2_velo + y_cnn1_velo + y_cnn2_velo) / 5).flatten()

    outcome_df_valid = pd.DataFrame({
        "predictions model 1": y_gru1_velo,
        "predictions model 2": y_gru2_velo,
        "predictions model 3": y_lstm_velo,
        "predictions model 4": y_cnn1_velo,
        "predictions model 5": y_cnn2_velo,
        "final prediction": y_pred_velo})
    outcome_df_valid['Standard deviation'] = outcome_df_valid[[
        "predictions model 1",
        "predictions model 2",
        "predictions model 3",
        "predictions model 4"]].std(axis=1)
    outcome_df_valid['Standard deviation %'] = (
        outcome_df_valid['Standard deviation']
        / outcome_df_valid['final prediction'] * 100
    )

    outcome_df_valid["abs_error"] = np.abs(outcome_df_valid["final prediction"] - y_velo) / y_velo

    valid_test_results_10 = outcome_df_valid[
        (outcome_df_valid["Standard deviation"] / outcome_df_valid["final prediction"] <= 0.1) &
        (outcome_df_valid["abs_error"] <= 0.1)
    ]
    valid_test_results_5 = outcome_df_valid[
        (outcome_df_valid["Standard deviation"] / outcome_df_valid["final prediction"] <= 0.1) &
        (outcome_df_valid["abs_error"] <= 0.05)
    ]
    filtered_outcome_df = outcome_df_valid[outcome_df_valid['Standard deviation %'] < 10]
    average_percentage_std = filtered_outcome_df['Standard deviation %'].mean()

    # Percentage of recovered bubbles with different std targets
    diff10_percent = len(valid_test_results_10) / (len(outcome_df_valid)) * 100
    diff05_percent = len(valid_test_results_5) / (len(outcome_df_valid)) * 100
    # Improvement in number of bubbles with a speed between AI and A2
    percent_improved = (
        ((valid_bubbles_ai - valid_bubbles_boring_software)
            / valid_bubbles_boring_software) * 100
    )
    # Percentage of original bubbles that the filtered ensemble predicts for
    percent_reported = len(filtered_outcome_df) / len(outcome_df_valid) * 100

    print(
        f"Percentage found valid bubbles (uncertainty < 10%) with speed "
        f"difference <10% from truth: {diff10_percent:.4f} %"
    )
    print(
        f"Percentage found valid bubbles (uncertainty < 10%) with speed "
        f"difference <5% from truth: {diff05_percent:.4f} %"
    )
    print(
        f"Percentage AI found valid bubbles (uncertainty < 10%): "
        f"{valid_bubbles_ai:.4f} % vs M2 analyzer: "
        f"{valid_bubbles_boring_software:.4f} %, improvement: "
        f"{percent_improved:.4f} %"
    )
    print(
        f"Model uncertainty (average uncertainty of valid bubbles): "
        f"{average_percentage_std:.4f} % with "
        f"{percent_reported:.2f} % of the labeled samples"
    )

    return outcome_df, outcome_df_valid, valid_test_results_10, y_velo

import traceback
from datetime import timedelta

import torch
import json
import pandas as pd
import numpy as np

from TimeSeries import TimeSeriesTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YellowUtesPredictor:

    def __init__(self, model_path, config_path, scaler_path=None):
        self.device = device
        self.config = self.load_config(config_path)
        self.model = self.load_model(model_path)
        self.feature_scaler = None
        self.target_scaler = None

        if scaler_path:
            self.load_scalers(scaler_path)

        # print(f"Model loaded successfully on {self.device}")
        # print(f"Model expects {self.config['sequence_length']} days of input")
        # print(f"Model predicts {self.config['prediction_length']} days ahead")

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:

            print(f"{config_path} not found.")
            print(traceback.format_exc())
            exit(1)

    def load_model(self, model_path):
        model = TimeSeriesTransformer(
            input_dim=self.config['input_dim'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            prediction_length=self.config['prediction_length']
        )

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def load_scalers(self, scaler_path):
        try:
            scaler_data = torch.load(scaler_path, map_location='cpu', weights_only=False)
            self.feature_scaler = scaler_data['feature_scaler']
            self.target_scaler = scaler_data['target_scaler']
        except FileNotFoundError:
            print(f"Scaler file not found: {scaler_path}")
            print("You'll need to provide scaling parameters manually")

    def prepare_input_data(self, data):

        # Ensure we have the required columns
        required_cols = self.config['feature_cols'] + [self.config['inference_target']]
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Take the last sequence_length days
        sequence_length = self.config['sequence_length']
        if len(data) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} days of data, got {len(data)}")

        data = data.tail(sequence_length).copy()

        # Extract features and target
        features = data[self.config['feature_cols']].values
        target = data[[self.config['inference_target']]].values

        # Scale the data
        if self.feature_scaler is None or self.target_scaler is None:
            # print("Warning: No scalers loaded. Using data as-is (not recommended)")
            scaled_features = features
            scaled_target = target
        else:
            scaled_features = self.feature_scaler.transform(features)
            scaled_target = self.target_scaler.transform(target)

        # Combine features and target
        input_sequence = np.concatenate([scaled_features, scaled_target], axis=1)

        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)

        return input_tensor

    def predict(self, data, return_confidence=False):
        with torch.no_grad():
            # Prepare input
            input_tensor = self.prepare_input_data(data)

            # Make prediction
            outputs = self.model(input_tensor)
            predictions_scaled = outputs['logits'].cpu().numpy()

            # Convert back to original scale
            if self.target_scaler is not None:
                predictions_original = self.target_scaler.inverse_transform(
                    predictions_scaled.reshape(-1, 1)
                ).flatten()
            else:
                predictions_original = predictions_scaled.flatten()
                # print("Warning: No target scaler found. Predictions may be in scaled format.")

            # Create prediction dates
            last_date = pd.to_datetime(data['date'].iloc[-1])
            prediction_dates = [
                last_date + timedelta(days=i + 1)
                for i in range(self.config['prediction_length'])
            ]

            # Format results
            results = {
                'predictions': predictions_original.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'input_period': f"{data['date'].iloc[0]} to {data['date'].iloc[-1]}",
                'prediction_period': f"{prediction_dates[0].strftime('%Y-%m-%d')} to {prediction_dates[-1].strftime('%Y-%m-%d')}",
                'model_config': self.config
            }

            return results

    def predict_from_csv(self):
        data = pd.read_csv("./datafiles/merged-dataset.csv")
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data = data.tail(self.config.get("sequence_length"))
        return self.predict(data)
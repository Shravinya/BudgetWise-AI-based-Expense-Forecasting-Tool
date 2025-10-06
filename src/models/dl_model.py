import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class DLModels:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "artifacts/models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, split_name: str = "train_engineered") -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Engineered file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded engineered data. Shape: {df.shape}")
        return df

    def prepare_dl_data(self, df: pd.DataFrame, sequence_length: int = 3):
        # Filter expenses, sort by date, use amount as target, features as input
        expense_df = df[df['transaction_type'] == 'Expense'].sort_values('date').reset_index(drop=True)
        if len(expense_df) < sequence_length + 1:
            raise ValueError("❌ Insufficient data for sequences.")
        
        features = ['amount', 'lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio']
        available_features = [f for f in features if f in expense_df.columns]
        if len(available_features) < 2:
            raise ValueError("❌ Insufficient features for DL models.")
        data = expense_df[available_features].fillna(0).values
        
        # Scale
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])
            y.append(data_scaled[i, 0])  # Predict next amount
        
        X, y = np.array(X), np.array(y)
        print(f"✅ Prepared DL data. X shape: {X.shape}, y shape: {y.shape}")
        
        # Save scaler
        scaler_path = self.output_dir / "dl_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return X, y, scaler

    def build_lstm(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def build_gru(self, input_shape):
        model = Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def build_bilstm(self, input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(50)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, model, X, y, epochs=50, batch_size=32):
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    def save_model(self, model, model_name: str):
        model_path = self.output_dir / f"{model_name}.h5"  # Keras saves as h5
        model.save(model_path)
        print(f"✅ {model_name} trained and saved: {model_path}")

    def initiate_dl(self, split_name: str = "train_engineered"):
        df = self.load_data(split_name)
        X, y, scaler = self.prepare_dl_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        
        # LSTM
        lstm_model = self.build_lstm(input_shape)
        lstm_model = self.train_model(lstm_model, X, y)
        self.save_model(lstm_model, "lstm")
        
        # GRU
        gru_model = self.build_gru(input_shape)
        gru_model = self.train_model(gru_model, X, y)
        self.save_model(gru_model, "gru")
        
        # Bi-LSTM
        bilstm_model = self.build_bilstm(input_shape)
        bilstm_model = self.train_model(bilstm_model, X, y)
        self.save_model(bilstm_model, "bi_lstm")
        
        print("✅ All DL models trained!")


if __name__ == "__main__":
    dl = DLModels()
    dl.initiate_dl()
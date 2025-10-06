import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluation:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "artifacts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def load_data(self, split_name: str = "val_engineered") -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Engineered file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {split_name} data. Shape: {df.shape}")
        return df

    def prepare_eval_data(self, df: pd.DataFrame):
        # Filter expenses
        expense_df = df[df['transaction_type'] == 'Expense'].sort_values('date').reset_index(drop=True)
        if len(expense_df) < 10:
            raise ValueError("❌ Insufficient data for evaluation.")
        
        # Features for ML and Baseline LR
        ml_features = ['lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio', 'inflation_rate', 'interest_rate', 'month', 'weekday']
        available_ml_features = [f for f in ml_features if f in expense_df.columns]
        X_ml = expense_df[available_ml_features].fillna(0)
        y_true = expense_df['amount'].values
        
        # Features specifically for Baseline LR
        lr_features = ['lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio']
        available_lr_features = [f for f in lr_features if f in expense_df.columns]
        X_lr = expense_df[available_lr_features].fillna(0)
        
        # For TS models, aggregate monthly
        if 'year' in expense_df.columns and 'month' in expense_df.columns:
            monthly_df = expense_df.groupby(['year', 'month'])['amount'].sum().reset_index()
            monthly_df['ds'] = pd.to_datetime(monthly_df['year'].astype(str) + '-' + 
                                              monthly_df['month'].astype(str) + '-01')
            ts_y_true = monthly_df['amount'].values
            ts_dates = monthly_df[['ds']]
            print(f"✅ Prepared eval data. Tabular shape: X_ml {X_ml.shape}, y {y_true.shape}; X_lr {X_lr.shape}; TS y {ts_y_true.shape}")
            return X_ml, X_lr, y_true, ts_y_true, ts_dates
        else:
            raise ValueError("❌ 'year' or 'month' column missing.")
        
        return X_ml, X_lr, y_true, None, None

    def compute_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
        # Directional accuracy (if more than 1 point)
        if len(y_true) > 1:
            dir_acc = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
        else:
            dir_acc = np.nan
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Directional Accuracy': dir_acc}

    def evaluate_baseline(self, X_ml, X_lr, y_true, ts_y_true, ts_dates):
        model_dir = self.input_dir / "models"
        
        # Linear Regression (tabular, use LR-specific features)
        if (model_dir / "linear_regression.pkl").exists():
            with open(model_dir / "linear_regression.pkl", 'rb') as f:
                lr_model = pickle.load(f)
            y_pred_lr = lr_model.predict(X_lr)
            metrics_lr = self.compute_metrics(y_true[:len(y_pred_lr)], y_pred_lr)
            metrics_lr['Model'] = 'Linear Regression'
            self.results.append(metrics_lr)
            print("✅ Linear Regression evaluated.")
        
        # ARIMA (TS, short-term forecast)
        if (model_dir / "arima.pkl").exists() and ts_y_true is not None and len(ts_y_true) > 10:
            with open(model_dir / "arima.pkl", 'rb') as f:
                arima_model = pickle.load(f)
            forecast_steps = min(3, len(ts_y_true) - len(arima_model.fittedvalues))  # Short-term 1 month
            if forecast_steps > 0:
                y_pred_arima = arima_model.forecast(steps=forecast_steps)
                metrics_arima_short = self.compute_metrics(ts_y_true[-forecast_steps:], y_pred_arima)
                metrics_arima_short['Model'] = 'ARIMA (Short-term)'
                self.results.append(metrics_arima_short)
                print("✅ ARIMA evaluated (short-term).")
        
        # Prophet (TS, mid-term forecast)
        if (model_dir / "prophet.pkl").exists() and ts_y_true is not None:
            try:
                from prophet import Prophet
                with open(model_dir / "prophet.pkl", 'rb') as f:
                    prophet_model = pickle.load(f)
                future = prophet_model.make_future_dataframe(periods=6, freq='MS')  # 6 months mid-term
                forecast = prophet_model.predict(future)
                y_pred_prophet = forecast['yhat'].tail(6).values  # Last 6 predictions
                metrics_prophet = self.compute_metrics(ts_y_true[-6:], y_pred_prophet)
                metrics_prophet['Model'] = 'Prophet (Mid-term)'
                self.results.append(metrics_prophet)
                print("✅ Prophet evaluated (mid-term).")
            except Exception as e:
                print(f"⚠️ Prophet evaluation skipped: {e}")

    def evaluate_ml(self, X_ml, y_true):
        model_dir = self.input_dir / "models"
        
        # Random Forest
        if (model_dir / "random_forest.pkl").exists():
            with open(model_dir / "random_forest.pkl", 'rb') as f:
                rf_model = pickle.load(f)
            y_pred_rf = rf_model.predict(X_ml)
            metrics_rf = self.compute_metrics(y_true, y_pred_rf)
            metrics_rf['Model'] = 'Random Forest'
            self.results.append(metrics_rf)
            print("✅ Random Forest evaluated.")
        
        # XGBoost
        if (model_dir / "xgboost.pkl").exists():
            with open(model_dir / "xgboost.pkl", 'rb') as f:
                xgb_model = pickle.load(f)
            y_pred_xgb = xgb_model.predict(X_ml)
            metrics_xgb = self.compute_metrics(y_true, y_pred_xgb)
            metrics_xgb['Model'] = 'XGBoost'
            self.results.append(metrics_xgb)
            print("✅ XGBoost evaluated.")
        
        # LightGBM
        if (model_dir / "lightgbm.pkl").exists():
            with open(model_dir / "lightgbm.pkl", 'rb') as f:
                lgb_model = pickle.load(f)
            y_pred_lgb = lgb_model.predict(X_ml)
            metrics_lgb = self.compute_metrics(y_true, y_pred_lgb)
            metrics_lgb['Model'] = 'LightGBM'
            self.results.append(metrics_lgb)
            print("✅ LightGBM evaluated.")

    def evaluate_dl(self, df):
        model_dir = self.input_dir / "models"
        sequence_length = 3
        features_dl = ['amount', 'lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio']
        available_dl = [f for f in features_dl if f in df.columns]
        expense_df = df[df['transaction_type'] == 'Expense'].sort_values('date').reset_index(drop=True)
        data_dl = expense_df[available_dl].fillna(0).values[-100:]  # Last 100 for eval
        if len(data_dl) < sequence_length + 1:
            print("⚠️ Insufficient data for DL evaluation.")
            return
        
        if (model_dir / "dl_scaler.pkl").exists():
            with open(model_dir / "dl_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler()
            scaler.fit(data_dl)
        
        data_scaled_dl = scaler.transform(data_dl)
        X_dl, y_true_dl = [], []
        for i in range(sequence_length, len(data_scaled_dl)):
            X_dl.append(data_scaled_dl[i-sequence_length:i])
            y_true_dl.append(data_scaled_dl[i, 0])
        X_dl, y_true_dl = np.array(X_dl), np.array(y_true_dl)
        y_true_dl_inv = scaler.inverse_transform(np.hstack((y_true_dl.reshape(-1,1), np.zeros((len(y_true_dl), len(available_dl)-1)))))[:, 0]
        
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.losses import MeanSquaredError
            custom_objects = {'mse': MeanSquaredError()}
        except ImportError:
            print("⚠️ TensorFlow not available for DL evaluation.")
            return
        
        # LSTM
        if (model_dir / "lstm.h5").exists():
            try:
                lstm_model = load_model(model_dir / "lstm.h5", custom_objects=custom_objects)
                y_pred_lstm_scaled = lstm_model.predict(X_dl, verbose=0)
                y_pred_lstm = scaler.inverse_transform(np.hstack((y_pred_lstm_scaled, np.zeros((len(y_pred_lstm_scaled), len(available_dl)-1)))))[:, 0]
                metrics_lstm = self.compute_metrics(y_true_dl_inv, y_pred_lstm)
                metrics_lstm['Model'] = 'LSTM'
                self.results.append(metrics_lstm)
                print("✅ LSTM evaluated.")
            except Exception as e:
                print(f"⚠️ LSTM evaluation skipped: {e}")
        
        # GRU
        if (model_dir / "gru.h5").exists():
            try:
                gru_model = load_model(model_dir / "gru.h5", custom_objects=custom_objects)
                y_pred_gru_scaled = gru_model.predict(X_dl, verbose=0)
                y_pred_gru = scaler.inverse_transform(np.hstack((y_pred_gru_scaled, np.zeros((len(y_pred_gru_scaled), len(available_dl)-1)))))[:, 0]
                metrics_gru = self.compute_metrics(y_true_dl_inv, y_pred_gru)
                metrics_gru['Model'] = 'GRU'
                self.results.append(metrics_gru)
                print("✅ GRU evaluated.")
            except Exception as e:
                print(f"⚠️ GRU evaluation skipped: {e}")
        
        # Bi-LSTM
        if (model_dir / "bi_lstm.h5").exists():
            try:
                bilstm_model = load_model(model_dir / "bi_lstm.h5", custom_objects=custom_objects)
                y_pred_bilstm_scaled = bilstm_model.predict(X_dl, verbose=0)
                y_pred_bilstm = scaler.inverse_transform(np.hstack((y_pred_bilstm_scaled, np.zeros((len(y_pred_bilstm_scaled), len(available_dl)-1)))))[:, 0]
                metrics_bilstm = self.compute_metrics(y_true_dl_inv, y_pred_bilstm)
                metrics_bilstm['Model'] = 'Bi-LSTM'
                self.results.append(metrics_bilstm)
                print("✅ Bi-LSTM evaluated.")
            except Exception as e:
                print(f"⚠️ Bi-LSTM evaluation skipped: {e}")

    def evaluate_transformers(self, df):
        model_dir = self.input_dir / "models"
        seq_len_trans = 12
        expense_df = df[df['transaction_type'] == 'Expense'].sort_values('date').reset_index(drop=True)
        amounts = expense_df['amount'].fillna(0).values[-50:].reshape(-1,1)  # Last 50 for eval
        if len(amounts) < seq_len_trans + 1:
            print("⚠️ Insufficient data for Transformer evaluation.")
            return
        
        if (model_dir / "transformer_scaler.pkl").exists():
            with open(model_dir / "transformer_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler()
            scaler.fit(amounts)
        
        amounts_scaled = scaler.transform(amounts)
        X_trans, y_true_trans = [], []
        for i in range(seq_len_trans, len(amounts_scaled)):
            X_trans.append(amounts_scaled[i-seq_len_trans:i])
            y_true_trans.append(amounts_scaled[i])
        X_trans, y_true_trans = np.array(X_trans), np.array(y_true_trans)
        y_true_trans_inv = scaler.inverse_transform(y_true_trans.reshape(-1,1)).flatten()
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("⚠️ PyTorch not installed. Skipping Transformer evaluation. Install with: pip install torch")
            return
        
        class SimpleTransformer(nn.Module):
            def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
                super().__init__()
                self.input_fc = nn.Linear(input_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_fc = nn.Linear(d_model, 1)
            
            def forward(self, x):
                x = self.input_fc(x)
                x = x.permute(1, 0, 2)  # For transformer
                x = self.transformer(x)
                x = x.mean(dim=0)  # Global avg pool
                return self.output_fc(x)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TFT Transformer
        if (model_dir / "tft_transformer.pth").exists():
            try:
                model = SimpleTransformer(input_dim=1)
                model.load_state_dict(torch.load(model_dir / "tft_transformer.pth", map_location=device))
                model.to(device)
                model.eval()
                
                dataset = TensorDataset(torch.tensor(X_trans, dtype=torch.float32))
                dataloader = DataLoader(dataset, batch_size=32)
                
                y_pred_scaled = []
                with torch.no_grad():
                    for batch_x in dataloader:
                        batch_x = batch_x[0].to(device)
                        output = model(batch_x)
                        y_pred_scaled.append(output.cpu().numpy())
                y_pred_scaled = np.concatenate(y_pred_scaled).flatten()
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                metrics_tft = self.compute_metrics(y_true_trans_inv, y_pred)
                metrics_tft['Model'] = 'TFT Transformer'
                self.results.append(metrics_tft)
                print("✅ TFT Transformer evaluated.")
            except Exception as e:
                print(f"⚠️ TFT Transformer evaluation skipped: {e}")
        
        # N-BEATS Transformer (similar)
        if (model_dir / "n_beats_transformer.pth").exists():
            try:
                model = SimpleTransformer(input_dim=1)
                model.load_state_dict(torch.load(model_dir / "n_beats_transformer.pth", map_location=device))
                model.to(device)
                model.eval()
                
                dataset = TensorDataset(torch.tensor(X_trans, dtype=torch.float32))
                dataloader = DataLoader(dataset, batch_size=32)
                
                y_pred_scaled = []
                with torch.no_grad():
                    for batch_x in dataloader:
                        batch_x = batch_x[0].to(device)
                        output = model(batch_x)
                        y_pred_scaled.append(output.cpu().numpy())
                y_pred_scaled = np.concatenate(y_pred_scaled).flatten()
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                metrics_nbeats = self.compute_metrics(y_true_trans_inv, y_pred)
                metrics_nbeats['Model'] = 'N-BEATS Transformer'
                self.results.append(metrics_nbeats)
                print("✅ N-BEATS Transformer evaluated.")
            except Exception as e:
                print(f"⚠️ N-BEATS Transformer evaluation skipped: {e}")

    def save_results(self):
        if self.results:
            results_df = pd.DataFrame(self.results)
            output_file = self.output_dir / "evaluation_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"✅ Evaluation results saved: {output_file}")
            
            # Select best model (lowest MAE)
            best_idx = results_df['MAE'].idxmin()
            best_model = results_df.iloc[best_idx]
            print(f"🏆 Best Model: {best_model['Model']} with MAE: {best_model['MAE']:.2f}")
        else:
            print("⚠️ No results to save.")

    def initiate_evaluation(self, split_name: str = "val_engineered"):
        df = self.load_data(split_name)
        X_ml, X_lr, y_true, ts_y_true, ts_dates = self.prepare_eval_data(df)
        
        self.evaluate_baseline(X_ml, X_lr, y_true, ts_y_true, ts_dates)
        self.evaluate_ml(X_ml, y_true)
        self.evaluate_dl(df)
        self.evaluate_transformers(df)
        
        self.save_results()
        print("✅ Full evaluation completed!")


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.initiate_evaluation()
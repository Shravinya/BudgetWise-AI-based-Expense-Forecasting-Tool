import os
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
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

    def prepare_ts_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter expenses
        expense_df = df[df['transaction_type'] == 'Expense']
        if len(expense_df) == 0:
            raise ValueError("❌ No expense data available.")
        
        # Aggregate to yearly-monthly total spend (since 'month' is 1-12, combine with 'year')
        if 'year' in expense_df.columns and 'month' in expense_df.columns:
            monthly_df = expense_df.groupby(['year', 'month'])['amount'].sum().reset_index()
            monthly_df['ds'] = pd.to_datetime(monthly_df['year'].astype(str) + '-' + 
                                              monthly_df['month'].astype(str) + '-01')
            monthly_df = monthly_df.rename(columns={'amount': 'y'})
            monthly_df = monthly_df[['ds', 'y']].sort_values('ds')
        else:
            raise ValueError("❌ 'year' or 'month' column missing for TS aggregation.")
        
        print(f"✅ Prepared monthly TS data. Shape: {monthly_df.shape}")
        return monthly_df

    def train_linear_regression(self, df: pd.DataFrame):
        # Simple linear reg on lag features (non-TS baseline)
        features = ['lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio']
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError("❌ No features available for Linear Regression.")
        X = df[available_features].dropna()
        y = df['amount'].iloc[:len(X)]
        model = LinearRegression()
        model.fit(X, y)
        
        model_path = self.output_dir / "linear_regression.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Linear Regression trained and saved: {model_path}")
        return model

    def train_arima(self, ts_df: pd.DataFrame):
        if len(ts_df) < 10:
            raise ValueError("❌ Insufficient data for ARIMA.")
        # ARIMA on monthly y
        model = ARIMA(ts_df['y'], order=(1,1,1))  # Simple order
        fitted_model = model.fit()
        
        model_path = self.output_dir / "arima.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"✅ ARIMA trained and saved: {model_path}")
        return fitted_model

    def train_prophet(self, ts_df: pd.DataFrame):
        if len(ts_df) < 2:
            raise ValueError("❌ Insufficient data for Prophet.")
        model = Prophet()
        model.fit(ts_df)
        
        model_path = self.output_dir / "prophet.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Prophet trained and saved: {model_path}")
        return model

    def initiate_baseline(self, split_name: str = "train_engineered"):
        df = self.load_data(split_name)
        ts_df = self.prepare_ts_data(df)
        
        self.train_linear_regression(df)
        self.train_arima(ts_df)
        self.train_prophet(ts_df)
        print("✅ All baseline models trained!")


if __name__ == "__main__":
    baseline = BaselineModels()
    baseline.initiate_baseline()
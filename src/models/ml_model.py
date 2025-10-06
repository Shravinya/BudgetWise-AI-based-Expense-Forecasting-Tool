import os
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

class MLModels:
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

    def prepare_ml_data(self, df: pd.DataFrame):
        # Filter expenses and use engineered features to predict 'amount'
        expense_df = df[df['transaction_type'] == 'Expense'].copy()
        if len(expense_df) == 0:
            raise ValueError("❌ No expense data available.")
        
        features = ['lag_1_month_spend', 'rolling_3m_avg', 'expense_ratio', 'inflation_rate', 'interest_rate', 'month', 'weekday']
        available_features = [f for f in features if f in expense_df.columns]
        if len(available_features) < 2:
            raise ValueError("❌ Insufficient features for ML models.")
        X = expense_df[available_features].fillna(0)
        y = expense_df['amount']
        
        print(f"✅ Prepared ML data. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train_random_forest(self, X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        model_path = self.output_dir / "random_forest.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Random Forest trained and saved: {model_path}")
        return model

    def train_xgboost(self, X, y):
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        model_path = self.output_dir / "xgboost.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ XGBoost trained and saved: {model_path}")
        return model

    def train_lightgbm(self, X, y):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X, y)
        
        model_path = self.output_dir / "lightgbm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ LightGBM trained and saved: {model_path}")
        return model

    def initiate_ml(self, split_name: str = "train_engineered"):
        df = self.load_data(split_name)
        X, y = self.prepare_ml_data(df)
        
        self.train_random_forest(X, y)
        self.train_xgboost(X, y)
        self.train_lightgbm(X, y)
        print("✅ All ML models trained!")


if __name__ == "__main__":
    ml = MLModels()
    ml.initiate_ml()
import os
import pandas as pd
from pathlib import Path

class FeatureEngineering:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "artifacts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_processed(self, split_name: str) -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}_processed.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Processed file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded processed {split_name}. Shape: {df.shape}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        # Lag features
        df['lag_1_month_spend'] = df['amount'].shift(1).fillna(0)

        # Rolling averages
        df['rolling_3m_avg'] = df['amount'].rolling(window=3).mean().fillna(0)

        # Expense ratios using 'Expense' for transaction_type
        if 'transaction_type' in df.columns:
            df['is_expense'] = (df['transaction_type'] == 'Expense').astype(int)
            total_income = df[df['is_expense'] == 0]['amount'].sum()
            if total_income > 0:
                df['expense_ratio'] = (df['amount'] * df['is_expense']) / total_income
            else:
                df['expense_ratio'] = 0
        else:
            df['expense_ratio'] = 0

        # External indicators (placeholders)
        df['inflation_rate'] = 0.05
        df['interest_rate'] = 0.07

        print("✅ Feature engineering completed.")
        return df

    def save_engineered(self, df: pd.DataFrame, split_name: str):
        output_file = self.output_dir / f"{split_name}_engineered.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Engineered data saved: {output_file}")

    def initiate_engineering(self, split_names: list = ["train", "val", "test"]):
        for split_name in split_names:
            df = self.load_processed(split_name)
            engineered_df = self.engineer_features(df)
            self.save_engineered(engineered_df, split_name)
        print("✅ All splits engineered successfully!")


if __name__ == "__main__":
    engineer = FeatureEngineering()
    engineer.initiate_engineering()
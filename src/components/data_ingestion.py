import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_path: str, output_dir: str = "artifacts"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.train_data_path = os.path.join(output_dir, "train.csv")
        self.val_data_path = os.path.join(output_dir, "val.csv")
        self.test_data_path = os.path.join(output_dir, "test.csv")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ File not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        print(f"🧩 Columns found: {list(df.columns)}")
        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df = df.dropna(how="all")

        # Expected columns (matched to actual dataset)
        expected_cols = ['transaction_id', 'user_id', 'date', 'transaction_type', 'category', 'amount', 'payment_mode', 'location', 'notes']

        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            print(f"⚠️ Warning: Missing columns detected → {missing}")
            print("   Proceeding with available columns...")

        # Handle date & amount columns safely
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])

        print("✅ Data validation & cleanup completed.")
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)

        train_df.to_csv(self.train_data_path, index=False)
        val_df.to_csv(self.val_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)

        print("✅ Data successfully split & saved:")
        print(f"   Train → {train_df.shape}")
        print(f"   Val   → {val_df.shape}")
        print(f"   Test  → {test_df.shape}")

        return self.train_data_path, self.val_data_path, self.test_data_path

    def initiate_data_ingestion(self):
        df = self.load_data()
        df = self.validate_data(df)
        return self.split_data(df)


if __name__ == "__main__":
    DATA_PATH = r"C:\Users\SHRAVINYA\Desktop\budget_tool\data\budgetwise_finance_dataset.csv"
    ingestion = DataIngestion(data_path=DATA_PATH)
    ingestion.initiate_data_ingestion()
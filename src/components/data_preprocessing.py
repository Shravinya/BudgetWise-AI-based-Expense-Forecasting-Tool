import os
import pandas as pd
import numpy as np
import re
from pathlib import Path

class DataPreprocessing:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "artifacts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_split(self, split_name: str) -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Input file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {split_name} split. Shape: {df.shape}")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Parse dates and create features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['weekday'] = df['date'].dt.weekday

        # Categorize using 'location' as proxy for merchant (refine category if needed)
        if 'location' in df.columns and 'category' in df.columns:
            def refine_category(row):
                if pd.isna(row['location']):
                    return row['category']
                location_lower = str(row['location']).lower()
                if re.search(r'food|restaurant|cafe|dining', location_lower):
                    return 'Food'
                elif re.search(r'rent|housing|utility', location_lower):
                    return 'Rent/Utilities'
                elif re.search(r'travel|transport|uber|ola', location_lower):
                    return 'Travel'
                elif re.search(r'shop|amazon|flipkart|retail', location_lower):
                    return 'Shopping'
                else:
                    return row['category']  # Keep original
            df['category'] = df.apply(refine_category, axis=1)

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df = df.fillna('Unknown')

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle outliers (IQR for 'amount')
        if 'amount' in df.columns:
            Q1 = df['amount'].quantile(0.25)
            Q3 = df['amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['amount'] >= lower_bound) & (df['amount'] <= upper_bound)]

        print("✅ Preprocessing steps completed.")
        return df

    def save_processed(self, df: pd.DataFrame, split_name: str):
        output_file = self.output_dir / f"{split_name}_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Processed data saved: {output_file}")

    def initiate_preprocessing(self, split_names: list = ["train", "val", "test"]):
        for split_name in split_names:
            df = self.load_split(split_name)
            processed_df = self.preprocess(df)
            self.save_processed(processed_df, split_name)
        print("✅ All splits preprocessed successfully!")


if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    preprocessor.initiate_preprocessing()
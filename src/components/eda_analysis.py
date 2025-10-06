import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EDAAnalysis:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "features/outputs"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_processed(self, split_name: str = "train_processed") -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Processed file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded processed data. Shape: {df.shape}")
        return df

    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter expenses using 'Expense' for transaction_type
        expense_df = df[df['transaction_type'] == 'Expense'] if 'transaction_type' in df.columns else df

        summary = {
            'total_transactions': len(df),
            'total_amount': df['amount'].sum() if 'amount' in df.columns else 0,
            'avg_amount': df['amount'].mean() if 'amount' in df.columns else 0,
            'categories_count': df['category'].nunique() if 'category' in df.columns else 0,
            'expense_transactions': len(expense_df)
        }
        summary_df = pd.DataFrame([summary])
        summary_path = self.input_dir / "eda_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary stats saved: {summary_path}")
        return summary_df

    def plot_category_spend(self, expense_df: pd.DataFrame):
        if len(expense_df) == 0:
            print("⚠️ No expense data available for category spend plot.")
            return
        if 'category' in expense_df.columns and 'amount' in expense_df.columns:
            category_spend = expense_df.groupby('category')['amount'].sum()
            if not category_spend.empty:
                plt.figure(figsize=(8, 6))
                category_spend.plot(kind='pie', autopct='%1.1f%%')
                plt.title('Spending Breakdown by Category')
                plt.ylabel('')
                plt.savefig(self.output_dir / "category_spend_pie.png")
                plt.close()
                print("✅ Category spend pie chart saved.")
            else:
                print("⚠️ No data for category spend plot.")

    def plot_monthly_trends(self, expense_df: pd.DataFrame):
        if len(expense_df) == 0:
            print("⚠️ No expense data available for monthly trends plot.")
            return
        if 'month' in expense_df.columns and 'amount' in expense_df.columns:
            monthly_spend = expense_df.groupby('month')['amount'].sum()
            if not monthly_spend.empty:
                plt.figure(figsize=(10, 6))
                monthly_spend.plot(kind='bar')
                plt.title('Monthly Spending Trends')
                plt.xlabel('Month')
                plt.ylabel('Total Amount')
                plt.savefig(self.output_dir / "monthly_spend_bar.png")
                plt.close()
                print("✅ Monthly trends bar chart saved.")
            else:
                print("⚠️ No data for monthly trends plot.")

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.savefig(self.output_dir / "correlation_heatmap.png")
            plt.close()
            print("✅ Correlation heatmap saved.")
        else:
            print("⚠️ Insufficient numeric columns for correlation heatmap.")

    def initiate_eda(self, split_name: str = "train_processed"):
        df = self.load_processed(split_name)
        expense_df = df[df['transaction_type'] == 'Expense'] if 'transaction_type' in df.columns else df
        self.generate_summary(df)
        self.plot_category_spend(expense_df)
        self.plot_monthly_trends(expense_df)
        self.plot_correlation_heatmap(df)
        print("✅ EDA analysis completed! Check outputs in features/outputs/")


if __name__ == "__main__":
    eda = EDAAnalysis()
    eda.initiate_eda()
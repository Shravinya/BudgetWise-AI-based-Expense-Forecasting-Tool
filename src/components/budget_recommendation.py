import os
import pandas as pd
from pathlib import Path

class BudgetRecommendation:
    def __init__(self, input_dir: str = "artifacts", output_dir: str = "recommendations"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_engineered(self, split_name: str = "train_engineered") -> pd.DataFrame:
        input_file = self.input_dir / f"{split_name}.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"❌ Engineered file not found: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded engineered data. Shape: {df.shape}")
        return df

    def generate_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'category' not in df.columns or 'amount' not in df.columns:
            raise ValueError("❌ Missing 'category' or 'amount' columns.")

        # Filter expenses using 'Expense'
        expense_df = df[df['transaction_type'] == 'Expense'] if 'transaction_type' in df.columns else df

        if len(expense_df) == 0:
            print("⚠️ No expense data available for recommendations.")
            return pd.DataFrame()

        # Category spend summary
        category_spend = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)

        if category_spend.empty:
            print("⚠️ No category spend data available.")
            return pd.DataFrame()

        recommendations = []
        total_expense = category_spend.sum()
        for category, spend in category_spend.items():
            ratio = spend / total_expense if total_expense > 0 else 0
            if ratio > 0.3:
                save_amount = spend * 0.1
                rec = {
                    'category': category,
                    'current_spend': spend,
                    'percentage_of_total': ratio * 100,
                    'recommendation': f"Reduce {category} spending by 10% to save {save_amount:.2f}",
                    'target_budget': spend * 0.9
                }
            else:
                rec = {
                    'category': category,
                    'current_spend': spend,
                    'percentage_of_total': ratio * 100,
                    'recommendation': f"Maintain current {category} spending",
                    'target_budget': spend
                }
            recommendations.append(rec)

        recs_df = pd.DataFrame(recommendations)
        print("✅ Recommendations generated.")
        return recs_df

    def save_recommendations(self, recs_df: pd.DataFrame):
        output_file = self.output_dir / "budget_recommendations.csv"
        recs_df.to_csv(output_file, index=False)
        print(f"✅ Recommendations saved: {output_file}")

    def initiate_recommendations(self, split_name: str = "train_engineered"):
        df = self.load_engineered(split_name)
        recs_df = self.generate_recommendations(df)
        if not recs_df.empty:
            self.save_recommendations(recs_df)
        print("✅ Budget recommendation pipeline completed!")


if __name__ == "__main__":
    recommender = BudgetRecommendation()
    recommender.initiate_recommendations()
# 💰 Budget Tool

**Budget Tool** is a personal finance management application built with Python and Streamlit. It helps users analyze spending, forecast expenses, and optimize budgets with AI/ML models.  

### Features
- **Data Upload & Processing:** Upload CSV transaction files; automatic preprocessing and train/validation/test splits.  
- **Exploratory Data Analysis (EDA):** Interactive charts for spending trends, category breakdowns, and time-based analysis.  
- **Expense Forecasting:** Predict 3–12 month expenses using:
  - Baseline models: Prophet, ARIMA  
  - ML models: Random Forest, XGBoost, LightGBM  
  - DL models: LSTM, GRU, Bi-LSTM, Transformers  
- **Budget Recommendations:** AI-driven insights to reduce overspending and optimize financial planning.  
- **Model Evaluation:** Metrics include MAE, RMSE, MAPE, and Directional Accuracy.  
- **Deployment:** Dockerized for portability; Streamlit UI for easy interaction.  

### Tech Stack
- **Backend:** Python 3.9+, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, PyTorch  
- **Forecasting:** Prophet, Statsmodels (ARIMA)  
- **Frontend:** Streamlit, Plotly  
- **Utilities:** Logging, exceptions, YAML configuration  
- **Deployment:** Docker  

### Setup Instructions
1. **Clone Repository:**  
   ```bash
   git clone https://github.com/Shravinya/BudgetWise-AI-based-Expense-Forecasting-Tool.git
   cd BudgetWise-AI-based-Expense-Forecasting-Tool
   pip install -r requirements.txt
   streamlit run app.py





# 💰 Budget Tool MVP

AI-Powered Expense Forecasting & Budget Optimization

---

## 🚀 Overview
Budget Tool MVP is an end-to-end solution for personal finance management, expense forecasting, and budget recommendations. It leverages machine learning, deep learning, and time series models to analyze your transaction data and provide actionable insights via an interactive dashboard.

---

## 📦 Project Structure
```
├── app.py                        # Streamlit dashboard
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── setup.py                      # Automated setup & quickstart
├── quick_start.py                # One-command quickstart script
├── artifacts/                    # Logs, models, processed data
│   ├── logs/                     # Log files
│   ├── models/                   # Saved models
│   ├── *.csv                     # Processed datasets
├── data/                         # Raw and sample data
├── recommendations/              # Budget recommendations output
├── src/                          # Source code
│   ├── components/               # Data ingestion, preprocessing, feature engineering, recommendations
│   ├── evaluation/               # Model evaluation scripts
│   ├── models/                   # ML, DL, baseline models
│   ├── pipelines/                # Main pipeline scripts
│   ├── utils/                    # Logging, exceptions, file operations
```

---

## ✨ Key Highlights & Features
- **Automated Data Ingestion & Preprocessing**
- **Feature Engineering**: Lag features, rolling averages, ratios, external indicators
- **Baseline, ML, DL, and Advanced Time Series Models**: Linear Regression, ARIMA, Prophet, Random Forest, XGBoost, LightGBM, LSTM, GRU, Bi-LSTM, Transformers
- **Enhanced Evaluation Framework**: MAE, RMSE, MAPE, Directional Accuracy, category-wise validation
- **Budget Recommendations**: Actionable suggestions to optimize spending
- **Interactive Streamlit Dashboard**: Upload CSV, visualize EDA, trends, forecasts, recommendations
- **Quickstart & Setup Automation**: One-command setup and run
- **Extensible & Modular Codebase**: Easy to add new models, features, or data sources

---

## 🧠 Capabilities
- Multi-source data ingestion (bank, UPI, wallet, Kaggle)
- Unified transaction dataset: `user_id`, `date`, `category`, `merchant`, `amount`, `income/expense`, `demographics`
- Outlier handling, missing value imputation, currency normalization
- EDA: Category-wise spend, seasonal patterns, income correlation
- Model training, evaluation, and selection
- Budget optimization recommendations
- REST API endpoints (optional, for integration)
- Dockerized deployment (optional)

---

## 🏗️ Machine Learning Pipeline
- **Data Ingestion**: `src/components/data_ingestion.py`
- **Preprocessing**: `src/components/data_preprocessing.py`
- **Feature Engineering**: `src/components/feature_engineering.py`
- **Model Training**: `src/pipelines/model_pipeline.py`
- **Evaluation**: `src/evaluation/model_evaluation.py`
- **Recommendations**: `src/components/budget_recommendation.py`

---

## 📊 Enhanced Evaluation Framework
- Supports multiple metrics: MAE, RMSE, MAPE, Directional Accuracy
- Category-wise and time-wise validation
- Logs and results saved in `artifacts/logs/` and `artifacts/evaluation_results.csv`

---

## 🖥️ Interactive Dashboard Features
- File upload (CSV/Excel)
- EDA: Pie/bar charts, monthly/weekly trends
- Forecast visualization: Line charts, next 3-month prediction
- Budget recommendations: Actionable tips (e.g., "Reduce dining by 10% to save ₹X")
- Model comparison and performance summary

---

## 🛠️ Tech Stack
- **Backend**: Python 3.9+, pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, prophet, torch
- **Frontend**: Streamlit
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Docker, Streamlit Cloud, Heroku, AWS (optional)

---

## ⚡ Quickstart
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd budget_tool
   ```
2. **Run setup script**
   ```sh
   python setup.py --install-all
   ```
   - Installs all dependencies
   - Creates directories
   - Sets up sample data
   - Runs initial data processing
   - Generates quick_start.py

3. **Quick Start (Recommended)**
   ```sh
   python quick_start.py
   ```
   - Trains models
   - Evaluates models
   - Launches Streamlit dashboard

4. **Manual Steps**
   - Add your data to `data/`
   - Run ingestion, preprocessing, feature engineering, recommendations, model training, evaluation as needed

---

## 🏃 Installation & Commands
- **Install dependencies only**
  ```sh
  python setup.py --skip-data
  ```
- **Skip dependency installation**
  ```sh
  python setup.py --skip-deps
  ```
- **Install advanced ML/DL libraries**
  ```sh
  python setup.py --install-all
  ```
- **Run initial data processing manually**
  ```sh
  python src/components/data_ingestion.py
  python src/components/data_preprocessing.py
  python src/components/feature_engineering.py
  python src/components/budget_recommendation.py
  ```
- **Train models manually**
  ```sh
  python src/models/baseline.py
  python src/models/ml_model.py
  python src/models/dl_model.py
  python src/models/transformers.py  # (optional)
  ```
- **Evaluate models**
  ```sh
  python src/evaluation/model_evaluation.py
  ```
- **Launch dashboard**
  ```sh
  streamlit run app.py
  ```

---

## 📖 Application Usage Guide
1. **Add your transaction data**
   - Place CSV files in `data/` directory
   - Required columns: `date`, `amount`, `category`, `transaction_type`
2. **Run setup or quickstart**
   - See above for commands
3. **Explore dashboard**
   - Upload your data
   - View EDA, forecasts, recommendations
4. **Review logs and results**
   - Check `artifacts/logs/` for logs
   - Check `artifacts/evaluation_results.csv` for model results
5. **Customize configuration**
   - Edit `config.yaml` for model/data settings
6. **Extend or integrate**
   - Add new models, features, or connect via REST API

---

## 📚 Documentation & Support
- See `README.md` and `config.yaml` for full documentation
- For issues, check logs or open a GitHub issue
- Contributions welcome!

---

## 🎯 Future Enhancements
- Real-time data integration
- Anomaly detection
- Multi-user support
- Advanced transformer models

---

**Built with ❤️ for smarter budgeting!**

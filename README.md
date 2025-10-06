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


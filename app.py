import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import os
from pathlib import Path
import tempfile
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from components.data_ingestion import DataIngestion
from components.data_preprocessing import DataPreprocessing
from components.eda_analysis import EDAAnalysis
from components.budget_recommendation import BudgetRecommendation

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Budget Tool MVP", page_icon="💰", layout="wide")

def main():
    # Beautiful header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">💰 Budget Tool MVP</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Unlock Financial Insights & Smart Recommendations</p>', unsafe_allow_html=True)

    # Sidebar with beautiful navigation
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        page = st.selectbox("Choose a section", ["📁 Upload Data", "📊 EDA Dashboard", "🔮 Forecast", "💡 Recommendations"], 
                            help="Select a feature to explore")
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        if os.path.exists("artifacts/train_processed.csv"):
            df = pd.read_csv("artifacts/train_processed.csv")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Total Spend", f"${df['amount'].sum():,.2f}")
            with col3:
                st.metric("Categories", df['category'].nunique())

    if page == "📁 Upload Data":
        upload_data_section()
    elif page == "📊 EDA Dashboard":
        eda_dashboard()
    elif page == "🔮 Forecast":
        forecast_section()
    elif page == "💡 Recommendations":
        recommendations_section()

def upload_data_section():
    st.header("📁 Upload & Process Your Transaction Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your bank/transaction CSV (columns: date, amount, category, etc.)")
    with col2:
        if st.button("🔄 Process Data", type="primary"):
            if uploaded_file is not None:
                process_uploaded_file(uploaded_file)
            else:
                st.warning("Please upload a file first!")

    if st.session_state.get('data_processed', False):
        st.success("✅ Data processed successfully!")
        display_data_preview()

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Processing ingestion...")
    ingestion = DataIngestion(data_path=tmp_path, output_dir="temp_artifacts")
    ingestion.initiate_data_ingestion()
    progress_bar.progress(30)

    status_text.text("Preprocessing data...")
    preprocessor = DataPreprocessing(input_dir="temp_artifacts", output_dir="temp_artifacts")
    preprocessor.initiate_preprocessing()
    progress_bar.progress(70)

    status_text.text("Generating EDA...")
    eda = EDAAnalysis(input_dir="temp_artifacts", output_dir="temp_artifacts/features/outputs")
    eda.initiate_eda("train_processed")
    progress_bar.progress(100)

    st.session_state['data_processed'] = True
    os.unlink(tmp_path)
    # Copy to artifacts using shutil
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    for file in Path("temp_artifacts").glob("*.csv"):
        shutil.copy(str(file), str(artifacts_dir / file.name))
    # Clean up temp
    shutil.rmtree("temp_artifacts")

def display_data_preview():
    df = pd.read_csv("artifacts/train.csv")  # Use raw for preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transactions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Amount", f"${df['amount'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Categories", df['category'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)

def eda_dashboard():
    st.header("📊 Interactive EDA Dashboard")
    
    try:
        df = pd.read_csv("artifacts/train_processed.csv")
        expense_df = df[df['transaction_type'] == 'Expense'] if 'transaction_type' in df.columns else df
    except FileNotFoundError:
        st.error("❌ No processed data found. Upload and process data first.")
        return

    tab1, tab2, tab3 = st.tabs(["Category Breakdown", "Trends Over Time", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if 'category' in expense_df.columns and 'amount' in expense_df.columns:
                category_spend = expense_df.groupby('category')['amount'].sum().reset_index()
                fig_pie = px.pie(category_spend, values='amount', names='category', 
                                 title='Spending by Category', color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.subheader("Top Categories")
            top_cats = category_spend.nlargest(5, 'amount')
            st.bar_chart(top_cats.set_index('category')['amount'])

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if 'month' in df.columns and 'amount' in expense_df.columns:
                monthly_spend = expense_df.groupby('month')['amount'].sum().reset_index()
                fig_line = px.line(monthly_spend, x='month', y='amount', title='Monthly Trends',
                                   markers=True, color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(fig_line, use_container_width=True)
        with col2:
            if 'weekday' in df.columns and 'amount' in expense_df.columns:
                weekday_spend = expense_df.groupby('weekday')['amount'].sum().reset_index()
                weekday_spend['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig_bar = px.bar(weekday_spend, x='day_name', y='amount', title='Weekly Patterns')
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spend", f"${expense_df['amount'].sum():,.2f}")
        with col2:
            st.metric("Avg Transaction", f"${expense_df['amount'].mean():.2f}")
        with col3:
            st.metric("Max Spend", f"${expense_df['amount'].max():.2f}")
        with col4:
            st.metric("Transactions", len(expense_df))

def forecast_section():
    st.header("🔮 Smart Expense Forecasting")
    
    try:
        df = pd.read_csv("artifacts/train_engineered.csv")
        expense_df = df[df['transaction_type'] == 'Expense']
    except FileNotFoundError:
        st.error("❌ No engineered data found. Process data first.")
        return

    # Prepare TS data for Prophet
    if 'year' in expense_df.columns and 'month' in expense_df.columns:
        monthly_df = expense_df.groupby(['year', 'month'])['amount'].sum().reset_index()
        monthly_df['ds'] = pd.to_datetime(monthly_df['year'].astype(str) + '-' + monthly_df['month'].astype(str) + '-01')
        monthly_df = monthly_df.rename(columns={'amount': 'y'})
        ts_df = monthly_df[['ds', 'y']].sort_values('ds')
    else:
        st.error("Insufficient data for forecasting.")
        return

    # Forecast horizon selector
    horizon = st.slider("Forecast Horizon (months)", 1, 12, 3)

    # Train and forecast
    with st.spinner(f"Generating {horizon}-month forecast..."):
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(ts_df)
        future = model.make_future_dataframe(periods=horizon, freq='MS')
        forecast = model.predict(future)

    # Interactive forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df['ds'], y=ts_df['y'], mode='lines+markers', 
                             name='Historical Spend', line=dict(color='#2e7d32', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', 
                             name='Forecast', line=dict(color='#ff9800', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                             fill=None, mode='lines', line_color='rgba(255,152,0,0.2)', name='Uncertainty'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                             fill='tonexty', mode='lines', line_color='rgba(255,152,0,0.2)', name=''))
    fig.update_layout(title=f'Expense Forecast: Next {horizon} Months', 
                      xaxis_title='Date', yaxis_title='Total Spend ($)', 
                      hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("📋 Forecast Details")
    forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).round(2)
    forecast_summary.columns = ['Date', 'Predicted Spend', 'Lower Bound', 'Upper Bound']
    st.dataframe(forecast_summary, use_container_width=True)

def recommendations_section():
    st.header("💡 Personalized Budget Recommendations")
    
    try:
        recs_df = pd.read_csv("recommendations/budget_recommendations.csv")
    except FileNotFoundError:
        st.error("❌ No recommendations found. Run budget_recommendation.py first.")
        return

    # Recommendations table with styling
    st.subheader("📊 Your Recommendations")
    st.dataframe(recs_df.style.highlight_max(subset=['percentage_of_total'], color='lightcoral'), use_container_width=True)
    
    # High-risk highlights
    high_risk = recs_df[recs_df['percentage_of_total'] > 30]
    if not high_risk.empty:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.warning("⚠️ **Action Needed:** These categories exceed 30% of your total spend!")
        for _, row in high_risk.iterrows():
            with st.expander(f"📉 {row['category']} - {row['recommendation']}"):
                st.write(f"**Current:** ${row['current_spend']:.2f} ({row['percentage_of_total']:.1f}%)")
                st.write(f"**Suggested:** ${row['target_budget']:.2f}")
                st.success(f"💰 Potential Savings: ${row['current_spend'] - row['target_budget']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("🎉 Great job! No high-risk categories detected. Keep it up!")

    # Overall tip
    st.info("💡 **Pro Tip:** Review your spending monthly and adjust based on life changes like income or goals.")

if __name__ == "__main__":
    main()
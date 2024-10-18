import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from data_handling import (
    load_data, 
    calculate_kpis, 
    format_data, 
    handle_missing_values, 
    validate_data, 
    clean_data,
    get_sales_target,
    get_monthly_metrics,
    get_ytd_metrics,
    calculate_order_metrics
)
from visualizations import (
    create_sales_by_product_chart,
    create_sales_vs_target_chart,
    create_sales_by_region_chart,
    create_monthly_sales_trend_chart,
    create_sales_by_channel_pie_chart
)

# Set page configuration
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide")

# Title
# Custom CSS for aligning the title to the top left
st.sidebar.markdown(
    """
    <div style='text-align: left; font-size: 32px; font-weight: bold;'>
        Sales Performance Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

# Load Data
@st.cache_data
def get_data():
    df = load_data()  # Load data from CSV using load_data from data_handling.py
    df = handle_missing_values(df)  # Handle missing values with appropriate strategies
    df = clean_data(df)  # Clean the data (e.g., fix invalid values, ensure correct types)
    df = format_data(df)  # Format data to ensure proper types and structure
    return df

# Load the data
df = get_data()

# Validate the loaded data and store the results
validation_results = validate_data(df)

# Sidebar navigation
st.sidebar.header("Navigation")
options = ["1. Explanation of Indicators", "2. Enter Data", "3. Monthly Dashboard", "4. YTD Dashboard"]
selection = st.sidebar.radio("Go to", options)

# Sidebar filters
if selection in ["3. Monthly Dashboard", "4. YTD Dashboard"]:
    st.sidebar.header("Filter Options")
    countries = df['Country'].unique().tolist() if 'Country' in df.columns else []
    selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries)
    filtered_df = df[df['Country'].isin(selected_countries)] if 'Country' in df.columns else df
else:
    filtered_df = df

# Show content based on the selection
match selection:
    case '1. Explanation of Indicators':
        st.header("Explanation of Indicators")
        st.write("""
    **Key Performance Indicators (KPIs):**
    - **Total Sales:** Sum of all sales revenue.
    - **Gross Profit:** Total sales minus cost of goods sold.
    - **Days Sales Outstanding (DSO):** Average number of days to collect payment after a sale.
    - **Days Payable Outstanding (DPO):** Average number of days it takes to pay invoices.
    - **Days Inventory Outstanding (DIO):** Average number of days inventory is held before sale.
    """)
    
        # Display the validation results for the default data
        st.write("Validation Results for Default Data:")
        st.write(validation_results)

    case '2. Enter Data':
        st.header("Enter Data")
        st.write("Upload a new sales data CSV file.")
        uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])
        
            # Validate the uploaded data
            validation_results = validate_data(df)
            st.write("Validation Results for Uploaded Data:")
            st.write(validation_results)
            
            # Handle missing values and clean the data
            df = handle_missing_values(df)
            df = clean_data(df)
            df = format_data(df)

            st.success("Data uploaded and processed successfully.")

# Monthly Dashboard (Selection 3)
    case "3. Monthly Dashboard":
        st.header("Monthly Dashboard")

        # Get the monthly metrics
        monthly_metrics = get_monthly_metrics(filtered_df)
        order_metrics = calculate_order_metrics(filtered_df, period='monthly')
        last_month = datetime.now() - timedelta(days=30)
        
        # Format month for 'vs last month' string
        last_month_str = str(last_month)
        last_month_date = datetime.strptime(last_month_str.split()[0], '%Y-%m-%d')
        formatted_date = last_month_date.strftime('%b %y')

        # Display metrics in a 3x2 grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Sales 💰", f"${monthly_metrics['current_sales']:,.2f}", f"{monthly_metrics['sales_change']:.1f}% vs {formatted_date}")
        with col2:
            st.metric("Monthly Profit 📈", f"${monthly_metrics['current_profit']:,.2f}", f"{monthly_metrics['profit_change']:.1f}% vs {formatted_date}")
        with col3:
            st.metric("Monthly # of Orders 🛒", f"{order_metrics['current_orders']:,}", f"{order_metrics['order_change']:.1f}% vs {formatted_date}")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Monthly DSO 📅", f"{monthly_metrics['current_dso']:.2f} days", f"{monthly_metrics['dso_change']:.1f}% vs {formatted_date}")
        with col5:
            st.metric("Monthly DPO 🏦", f"{monthly_metrics['current_dpo']:.2f} days", f"{monthly_metrics['dpo_change']:.1f}% vs {formatted_date}")
        with col6:
            st.metric("Monthly DIO 📦", f"{monthly_metrics['current_dio']:.2f} days", f"{monthly_metrics['dio_change']:.1f}% vs {formatted_date}")

        # Filter data for the past month
        monthly_filtered_df = filtered_df[filtered_df['InvoiceDate'] >= last_month]

        # Check if there is data for the past month
        if not monthly_filtered_df.empty:
            # Visualizations in a 3x2 grid using the filtered data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_sales_by_product_chart(monthly_filtered_df), use_container_width=True)
                st.plotly_chart(create_sales_by_region_chart(monthly_filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(create_monthly_sales_trend_chart(monthly_filtered_df), use_container_width=True)
                st.plotly_chart(create_sales_vs_target_chart(monthly_filtered_df), use_container_width=True)
            with col3:
                st.plotly_chart(create_sales_by_channel_pie_chart(monthly_filtered_df), use_container_width=True)
                # Add another visualization here if needed
        else:
            st.warning("No data available for the past month.")

# YTD Dashboard (Selection 4)
    case "4. YTD Dashboard":
        st.header("Year-to-Date (YTD) Dashboard")

        # Get the YTD metrics
        ytd_metrics = get_ytd_metrics(filtered_df)
        ytd_order_metrics = calculate_order_metrics(filtered_df, period='ytd')
        last_year = datetime.now() - timedelta(days=365)

        # Format YTD for 'vs YTD' str
        last_year_str = str(last_year)
        last_year_date = datetime.strptime(last_year_str.split()[0], '%Y-%m-%d')
        formatted_year_date = last_year_date.strftime('%b %y')

        # Display metrics in a 3x2 grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Year to Date Sales 💰", f"${ytd_metrics['current_sales']:,.2f}", f"{ytd_metrics['sales_change']:.1f}% vs {formatted_year_date}")
        with col2:
            st.metric("Year to Date Profit 📈", f"${ytd_metrics['current_profit']:,.2f}", f"{ytd_metrics['profit_change']:.1f}% vs {formatted_year_date}")
        with col3:
            st.metric("Year to Date # of Orders 🛒", f"{ytd_order_metrics['current_orders']:,}", f"{ytd_order_metrics['order_change']:.1f}% vs {formatted_year_date}")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Year to Date DSO 📅", f"{ytd_metrics['current_dso']:.2f} days", f"{ytd_metrics['dso_change']:.1f}% vs {formatted_year_date}")
        with col5:
            st.metric("Year to Date DPO 🏦", f"{ytd_metrics['current_dpo']:.2f} days", f"{ytd_metrics['dpo_change']:.1f}% vs {formatted_year_date}")
        with col6:
            st.metric("Year to Date DIO 📦", f"{ytd_metrics['current_dio']:.2f} days", f"{ytd_metrics['dio_change']:.1f}% vs {formatted_year_date}")
        
        # Filter data for the last year
        ytd_filtered_df = filtered_df[filtered_df['InvoiceDate'] >= last_year]

        # Check if there is data for the last year
        if not ytd_filtered_df.empty:
            # Visualizations in a 3x2 grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(create_sales_by_product_chart(ytd_filtered_df), use_container_width=True)
                st.plotly_chart(create_sales_by_region_chart(ytd_filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(create_monthly_sales_trend_chart(ytd_filtered_df), use_container_width=True)
                st.plotly_chart(create_sales_vs_target_chart(ytd_filtered_df), use_container_width=True)
            with col3:
                st.plotly_chart(create_sales_by_channel_pie_chart(ytd_filtered_df), use_container_width=True)
                # Add another visualization here if needed
        else:
            st.warning("No data available for the last year.")

    case _: 
        st.warning("Invalid selection, exiting the app.")
        st.stop()

# Footer
footer = """
<style>
footer {
    visibility: hidden;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: left;
    padding: 10px;
    font-size: 12px;
}
</style>

<div class="footer">
    <p>© 2024 Angel Jimenez. All rights reserved.</p>
</div>
"""

# Add custom CSS to position the copyright notice at the bottom of the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100vh;
    }
    .footer {
        text-align: left;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a copyright notice at the bottom of the sidebar
st.sidebar.markdown(
    """
    <div class="footer">
        <small>&copy; 2024 Angel Jimenez. All rights reserved.</small>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide")

# Title
st.title("Sales Performance Dashboard")

# Load Data
@st.cache_data
def load_data():
    # Load the cleaned data from the CSV file
    df = pd.read_csv('data/sales_data.csv', parse_dates=['InvoiceDate'])
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Clean up the InvoiceNo column
    df['InvoiceNo'] = df['InvoiceNo'].str.replace('^[A-Za-z]+', '', regex=True)
    
    # Convert data types
    for column in df.columns:
        match column:
            case 'InvoiceNo' | 'Quantity' | 'UnitPrice' | 'CustomerID':
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].astype(int)
            case 'StockCode' | 'Description' | 'Country':
                if pd.api.types.is_string_dtype(df[column]):
                    df[column] = df[column].astype(str)
            case 'InvoiceDate':
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column])
    
    # Create the 'Sales' column
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Sales'] = df['Quantity'] * df['UnitPrice']
    
    return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
countries = df['Country'].unique().tolist() if 'Country' in df.columns else []
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries)

# Filter data based on selections
filtered_df = df[df['Country'].isin(selected_countries)] if 'Country' in df.columns else df

# Display key metrics
if 'Quantity' in filtered_df.columns:
    total_quantity = filtered_df['Quantity'].sum()
    st.metric("Total Quantity Sold", f"{total_quantity}")

if 'Sales' in filtered_df.columns:
    total_sales = filtered_df['Sales'].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")


# Visualizations
# Sales by Country
if 'Country' in filtered_df.columns and 'Sales' in filtered_df.columns:
    sales_by_country = filtered_df.groupby('Country')['Sales'].sum().reset_index()
    fig1 = px.bar(sales_by_country, x='Country', y='Sales', title='Sales by Country', color='Country')
    st.plotly_chart(fig1, use_container_width=True)

# Monthly Sales Trend
if 'InvoiceDate' in filtered_df.columns and 'Sales' in filtered_df.columns:
    monthly_sales = filtered_df.resample('ME', on='InvoiceDate')['Sales'].sum().reset_index()
    fig2 = px.line(monthly_sales, x='InvoiceDate', y='Sales', title='Monthly Sales')
    fig2.update_layout(xaxis_title='Date')
    st.plotly_chart(fig2, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide")

# Title
st.title("Sales Performance Dashboard")

# Load Data
@st.cache_data
def load_data():
    # Load the cleaned data from the CSV file
    df = pd.read_csv('data/sales_data.csv', parse_dates=['InvoiceDate'])
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Clean up the InvoiceNo column
    df['InvoiceNo'] = df['InvoiceNo'].str.replace('^[A-Za-z]+', '', regex=True)
    
    # Convert data types
    for column in df.columns:
        match column:
            case 'InvoiceNo' | 'Quantity' | 'UnitPrice' | 'CustomerID':
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].astype(int)
            case 'StockCode' | 'Description' | 'Country':
                if pd.api.types.is_string_dtype(df[column]):
                    df[column] = df[column].astype(str)
            case 'InvoiceDate':
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column])
    
    # Create the 'Sales' column
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Sales'] = df['Quantity'] * df['UnitPrice']
    
    return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
countries = df['Country'].unique().tolist() if 'Country' in df.columns else []
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries)

# Filter data based on selections
filtered_df = df[df['Country'].isin(selected_countries)] if 'Country' in df.columns else df

# Display key metrics
if 'Quantity' in filtered_df.columns:
    total_quantity = filtered_df['Quantity'].sum()
    st.metric("Total Quantity Sold", f"{total_quantity}")

if 'Sales' in filtered_df.columns:
    total_sales = filtered_df['Sales'].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")


# Visualizations
# Sales by Country
if 'Country' in filtered_df.columns and 'Sales' in filtered_df.columns:
    sales_by_country = filtered_df.groupby('Country')['Sales'].sum().reset_index()
    fig1 = px.bar(sales_by_country, x='Country', y='Sales', title='Sales by Country', color='Country')
    st.plotly_chart(fig1, use_container_width=True)

# Monthly Sales Trend
if 'InvoiceDate' in filtered_df.columns and 'Sales' in filtered_df.columns:
    monthly_sales = filtered_df.resample('ME', on='InvoiceDate')['Sales'].sum().reset_index()
    fig2 = px.line(monthly_sales, x='InvoiceDate', y='Sales', title='Monthly Sales')
    fig2.update_layout(xaxis_title='Date')
    st.plotly_chart(fig2, use_container_width=True)

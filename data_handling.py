# Imports
import pandas as pd
from fuzzywuzzy import process
from datetime import datetime, timedelta

# Configurations and Constants
DEFAULT_DATA_PATH = 'data/sales_data.csv'

with open('countries.txt', 'r') as file:
    VALID_COUNTRIES = [line.strip() for line in file]

# Dictionary for common misspellings or abbreviations
KNOWN_MISTAKES = {
    'Uk': 'United Kingdom',
    'Usa': 'United States',
    'Misstake': 'United States' 
}

DATE_COLUMN = 'InvoiceDate'
PRODUCT_COLUMN = 'Product'
QUANTITY_COLUMN = 'Quantity'
UNIT_PRICE_COLUMN = 'UnitPrice'
SALES_COLUMN = 'Sales'
COUNTRY_COLUMN = 'Country'
CHANNEL_COLUMN = 'Channel'
COGS_COLUMN = 'COGS'
ACCOUNTS_RECEIVABLE_COLUMN = 'AccountsReceivable'
ACCOUNTS_PAYABLE_COLUMN = 'AccountsPayable'
INVENTORY_COLUMN = 'Inventory'
SALES_TARGET_COLUMN = 'SalesTarget'

# Data loading functions
def load_data(file_path=DEFAULT_DATA_PATH):
    """
    Load the sales data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=[DATE_COLUMN])
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found


# Data validation
def validate_data(df):
    """
    Validate the data for common issues like missing values, invalid types, and unrealistic values.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'missing_values': df.isnull().sum(),
        'invalid_data_types': {},
        'negative_values': {},
        'invalid_dates': 0,
        'outliers': {}
    }

    # Check for invalid data types
    for column in [QUANTITY_COLUMN, UNIT_PRICE_COLUMN, COGS_COLUMN,
                   ACCOUNTS_RECEIVABLE_COLUMN, ACCOUNTS_PAYABLE_COLUMN,
                   INVENTORY_COLUMN, SALES_TARGET_COLUMN]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            validation_results['invalid_data_types'][column] = 'Non-numeric values detected'

    # Check for negative values, but only for numeric columns
    for column in [QUANTITY_COLUMN, UNIT_PRICE_COLUMN, COGS_COLUMN,
                   ACCOUNTS_RECEIVABLE_COLUMN, ACCOUNTS_PAYABLE_COLUMN,
                   INVENTORY_COLUMN]:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Count how many negative values exist
            validation_results['negative_values'][column] = (df[column] < 0).sum()

    # Check for invalid dates
    validation_results['invalid_dates'] = df[DATE_COLUMN].apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull().sum()

    # Detect outliers using the interquartile range (IQR) method for numeric columns
    for column in [QUANTITY_COLUMN, UNIT_PRICE_COLUMN, COGS_COLUMN,
                   ACCOUNTS_RECEIVABLE_COLUMN, ACCOUNTS_PAYABLE_COLUMN,
                   INVENTORY_COLUMN, SALES_TARGET_COLUMN]:
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
            validation_results['outliers'][column] = outliers

    return validation_results


def handle_missing_values(df):
    """
    Handle missing values in the DataFrame with appropriate strategies.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    for column in df.columns:
        # Coerce to numeric where applicable
        if pd.api.types.is_numeric_dtype(df[column]):
            # Convert to numeric, setting errors to NaN
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # Fill missing values based on the percentage of missing values
            if df[column].isnull().mean() > 0.1:
                df[column] = df[column].fillna(0)
            else:
                df[column] = df[column].fillna(df[column].mean())
        else:
            # For non-numeric columns, fill with mode or 'Unknown'
            df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
    return df

# Data cleaning/formatting
def format_data(df):
    """
    Format the data types in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to format.

    Returns:
        pd.DataFrame: Formatted DataFrame.
    """
    # Convert InvoiceDate to datetime if not already
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Ensure numerical columns are correctly formatted
    numeric_columns = [
        QUANTITY_COLUMN, UNIT_PRICE_COLUMN, SALES_COLUMN, COGS_COLUMN,
        ACCOUNTS_RECEIVABLE_COLUMN, ACCOUNTS_PAYABLE_COLUMN, INVENTORY_COLUMN,
        SALES_TARGET_COLUMN
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)

    # Clean up text fields, e.g., Product and Country
    text_columns = [PRODUCT_COLUMN, COUNTRY_COLUMN, CHANNEL_COLUMN]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].str.strip().str.title()

    return df


def clean_data(df):
    """
    Clean the DataFrame by fixing data errors, handling invalid values, and ensuring correct types.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Fix negative values by setting them to zero
    for column in [QUANTITY_COLUMN, UNIT_PRICE_COLUMN, COGS_COLUMN,
                   ACCOUNTS_RECEIVABLE_COLUMN, ACCOUNTS_PAYABLE_COLUMN,
                   INVENTORY_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors='coerce').apply(lambda x: max(x, 0))

    # Correct invalid dates
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce').fillna(pd.Timestamp('2000-01-01'))

    # Ensure data types are correct
    df = format_data(df)

    # Add derived columns if missing
    df = add_sales_column(df)

    # Clean country names (filter)
    df = clean_country_names(df)

    return df


def clean_country_names(df, column_name='Country'):
    """
    Clean the 'Country' column in the DataFrame by fixing common misspellings.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'Country' column.
        column_name (str): The name of the column to clean. Default is 'Country'.

    Returns:
        pd.DataFrame: DataFrame with cleaned country names.
    """
    def get_closest_country(name):
        # Check if the name is already in VALID_COUNTRIES
        if name in VALID_COUNTRIES:
            return name
        
        # Check for known misspellings or abbreviations
        if name in KNOWN_MISTAKES:
            return KNOWN_MISTAKES[name]
        
        # If the name is missing or not a string, return NaN
        if pd.isna(name) or not isinstance(name, str):
            return np.nan
        
        # Use fuzzywuzzy to find the closest match with a high score cutoff
        match = process.extractOne(name, VALID_COUNTRIES, score_cutoff=90)
        
        if match:
            # If a close match is found, return the closest match
            return match[0]
        else:
            # If no match is found, set the value to NaN
            return np.nan

    # Apply the cleaning function to the specified column
    df[column_name] = df[column_name].apply(get_closest_country)
    
    return df


# KPI Calculation functions
def calculate_total_sales(df):
    """
    Calculate the total sales.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        float: Total sales value.
    """
    # Coerce Sales column to numeric
    df[SALES_COLUMN] = pd.to_numeric(df[SALES_COLUMN], errors='coerce').fillna(0)
    return df[SALES_COLUMN].sum()


def calculate_gross_profit(df):
    """
    Calculate the gross profit.

    Args:
        df (pd.DataFrame): The DataFrame containing sales and COGS data.

    Returns:
        float: Gross profit value.
    """
    # Coerce Sales and COGS columns to numeric
    df[SALES_COLUMN] = pd.to_numeric(df[SALES_COLUMN], errors='coerce').fillna(0)
    df[COGS_COLUMN] = pd.to_numeric(df[COGS_COLUMN], errors='coerce').fillna(0)
    return df[SALES_COLUMN].sum() - df[COGS_COLUMN].sum()


def calculate_days_sales_outstanding(df):
    """
    Calculate Days Sales Outstanding (DSO).

    Args:
        df (pd.DataFrame): The DataFrame containing sales and accounts receivable data.

    Returns:
        float: DSO value.
    """
    # Coerce relevant columns to numeric
    df[SALES_COLUMN] = pd.to_numeric(df[SALES_COLUMN], errors='coerce').fillna(0)
    df[ACCOUNTS_RECEIVABLE_COLUMN] = pd.to_numeric(df[ACCOUNTS_RECEIVABLE_COLUMN], errors='coerce').fillna(0)
    
    total_sales = df[SALES_COLUMN].sum()
    accounts_receivable = df[ACCOUNTS_RECEIVABLE_COLUMN].sum()
    
    if total_sales > 0:
        return (accounts_receivable / total_sales) * 365
    return 0


def calculate_days_payable_outstanding(df):
    """
    Calculate Days Payable Outstanding (DPO).

    Args:
        df (pd.DataFrame): The DataFrame containing COGS and accounts payable data.

    Returns:
        float: DPO value.
    """
    # Coerce relevant columns to numeric
    df[COGS_COLUMN] = pd.to_numeric(df[COGS_COLUMN], errors='coerce').fillna(0)
    df[ACCOUNTS_PAYABLE_COLUMN] = pd.to_numeric(df[ACCOUNTS_PAYABLE_COLUMN], errors='coerce').fillna(0)
    
    cogs_total = df[COGS_COLUMN].sum()
    accounts_payable = df[ACCOUNTS_PAYABLE_COLUMN].sum()
    
    if cogs_total > 0:
        return (accounts_payable / cogs_total) * 365
    return 0


def calculate_days_inventory_outstanding(df):
    """
    Calculate Days Inventory Outstanding (DIO).

    Args:
        df (pd.DataFrame): The DataFrame containing inventory and COGS data.

    Returns:
        float: DIO value.
    """
    # Coerce relevant columns to numeric
    df[COGS_COLUMN] = pd.to_numeric(df[COGS_COLUMN], errors='coerce').fillna(0)
    df[INVENTORY_COLUMN] = pd.to_numeric(df[INVENTORY_COLUMN], errors='coerce').fillna(0)
    
    cogs_total = df[COGS_COLUMN].sum()
    inventory = df[INVENTORY_COLUMN].sum()
    
    if cogs_total > 0:
        return (inventory / cogs_total) * 365
    return 0


def calculate_kpis(df):
    """
    Calculate all relevant KPIs for the sales dashboard.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        dict: Dictionary containing calculated KPI values.
    """
    total_sales = calculate_total_sales(df)
    gross_profit = calculate_gross_profit(df)
    dso = calculate_days_sales_outstanding(df)
    dpo = calculate_days_payable_outstanding(df)
    dio = calculate_days_inventory_outstanding(df)

    kpis = {
        'Total Sales': total_sales,
        'Gross Profit': gross_profit,
        'Days Sales Outstanding': dso,
        'Days Payable Outstanding': dpo,
        'Days Inventory Outstanding': dio
    }
    return kpis

def get_sales_target(df):
    """
    Extract the SalesTarget column from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        list: List of sales target values.
    """
    if SALES_TARGET_COLUMN in df.columns:
        sales_target = df[SALES_TARGET_COLUMN].tolist()
    else:
        # If SalesTarget column does not exist, return a list of zeros as a placeholder
        sales_target = [0] * len(df)
    return sales_target


# Helper function to calculate percentage change
def calculate_percentage_change(current, previous):
    """
    Calculate the percentage change between the current and previous values.
    
    Args:
        current (float): The current month's value.
        previous (float): The previous month's value.

    Returns:
        float: The percentage change from the previous to the current value.
    """
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def get_monthly_metrics(df):
    """
    Calculate monthly metrics for Sales, Profit, DSO, DPO, and DIO.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
    
    Returns:
        dict: Dictionary containing current and previous month metrics and their percentage changes.
    """
    # Resample data to monthly frequency and calculate metrics
    monthly_data = df.resample('ME', on=DATE_COLUMN).agg({
        SALES_COLUMN: 'sum',
        COGS_COLUMN: 'sum',
        ACCOUNTS_RECEIVABLE_COLUMN: 'sum',
        ACCOUNTS_PAYABLE_COLUMN: 'sum',
        INVENTORY_COLUMN: 'sum'
    }).reset_index()

    # Get current and previous month metrics
    current_month = monthly_data.iloc[-1] if len(monthly_data) > 0 else None
    previous_month = monthly_data.iloc[-2] if len(monthly_data) > 1 else None

    # Calculate monthly sales and profit
    current_sales = current_month[SALES_COLUMN] if current_month is not None else 0
    previous_sales = previous_month[SALES_COLUMN] if previous_month is not None else 0
    sales_change = calculate_percentage_change(current_sales, previous_sales)

    current_profit = current_sales - current_month[COGS_COLUMN] if current_month is not None else 0
    previous_profit = previous_sales - previous_month[COGS_COLUMN] if previous_month is not None else 0
    profit_change = calculate_percentage_change(current_profit, previous_profit)

    # Calculate DSO, DPO, and DIO
    current_dso = (current_month[ACCOUNTS_RECEIVABLE_COLUMN] / current_sales * 30) if current_sales > 0 else 0
    previous_dso = (previous_month[ACCOUNTS_RECEIVABLE_COLUMN] / previous_sales * 30) if previous_sales > 0 else 0
    dso_change = calculate_percentage_change(current_dso, previous_dso)

    current_dpo = (current_month[ACCOUNTS_PAYABLE_COLUMN] / current_month[COGS_COLUMN] * 30) if current_month[COGS_COLUMN] > 0 else 0
    previous_dpo = (previous_month[ACCOUNTS_PAYABLE_COLUMN] / previous_month[COGS_COLUMN] * 30) if previous_month[COGS_COLUMN] > 0 else 0
    dpo_change = calculate_percentage_change(current_dpo, previous_dpo)

    current_dio = (current_month[INVENTORY_COLUMN] / current_month[COGS_COLUMN] * 30) if current_month[COGS_COLUMN] > 0 else 0
    previous_dio = (previous_month[INVENTORY_COLUMN] / previous_month[COGS_COLUMN] * 30) if previous_month[COGS_COLUMN] > 0 else 0
    dio_change = calculate_percentage_change(current_dio, previous_dio)

    # Return the metrics in a dictionary
    return {
        'current_sales': current_sales,
        'previous_sales': previous_sales,
        'sales_change': sales_change,
        'current_profit': current_profit,
        'previous_profit': previous_profit,
        'profit_change': profit_change,
        'current_dso': current_dso,
        'previous_dso': previous_dso,
        'dso_change': dso_change,
        'current_dpo': current_dpo,
        'previous_dpo': previous_dpo,
        'dpo_change': dpo_change,
        'current_dio': current_dio,
        'previous_dio': previous_dio,
        'dio_change': dio_change
    }


def calculate_order_metrics(df, period='monthly'):
    """
    Calculate the number of orders and their percent change.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
        period (str): Either 'monthly' or 'ytd' to specify the time frame.

    Returns:
        dict: Dictionary containing the number of orders and percent change.
    """
    if period == 'monthly':
        # Filter data for the past month
        last_month = datetime.now() - timedelta(days=30)
        current_period_df = df[df[DATE_COLUMN] >= last_month]
        
        # Filter data for the previous month
        previous_month = last_month - timedelta(days=30)
        previous_period_df = df[(df[DATE_COLUMN] < last_month) & (df[DATE_COLUMN] >= previous_month)]
    elif period == 'ytd':
        # Filter data for the current year
        current_year = datetime.now().year
        current_period_df = df[df[DATE_COLUMN].dt.year == current_year]

        # Filter data for the previous year
        previous_year = current_year - 1
        previous_period_df = df[df[DATE_COLUMN].dt.year == previous_year]
    else:
        return None

    # Calculate number of orders
    current_orders = current_period_df.shape[0]
    previous_orders = previous_period_df.shape[0]
    order_change = calculate_percentage_change(current_orders, previous_orders)

    return {
        'current_orders': current_orders,
        'previous_orders': previous_orders,
        'order_change': order_change
    }


def get_ytd_metrics(df):
    """
    Calculate YTD metrics for Sales, Profit, DSO, DPO, and DIO.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
    
    Returns:
        dict: Dictionary containing current and previous YTD metrics and their percentage changes.
    """
    # Resample data to yearly frequency and calculate metrics
    ytd_data = df.resample('YE', on=DATE_COLUMN).agg({
        SALES_COLUMN: 'sum',
        COGS_COLUMN: 'sum',
        ACCOUNTS_RECEIVABLE_COLUMN: 'sum',
        ACCOUNTS_PAYABLE_COLUMN: 'sum',
        INVENTORY_COLUMN: 'sum'
    }).reset_index()

    # Get current and previous year metrics
    current_year = ytd_data.iloc[-1] if len(ytd_data) > 0 else None
    previous_year = ytd_data.iloc[-2] if len(ytd_data) > 1 else None

    # Calculate YTD sales and profit
    current_sales = current_year[SALES_COLUMN] if current_year is not None else 0
    previous_sales = previous_year[SALES_COLUMN] if previous_year is not None else 0
    sales_change = calculate_percentage_change(current_sales, previous_sales)

    current_profit = current_sales - current_year[COGS_COLUMN] if current_year is not None else 0
    previous_profit = previous_sales - previous_year[COGS_COLUMN] if previous_year is not None else 0
    profit_change = calculate_percentage_change(current_profit, previous_profit)

    # Calculate DSO, DPO, and DIO
    current_dso = (current_year[ACCOUNTS_RECEIVABLE_COLUMN] / current_sales * 365) if current_sales > 0 else 0
    previous_dso = (previous_year[ACCOUNTS_RECEIVABLE_COLUMN] / previous_sales * 365) if previous_sales > 0 else 0
    dso_change = calculate_percentage_change(current_dso, previous_dso)

    current_dpo = (current_year[ACCOUNTS_PAYABLE_COLUMN] / current_year[COGS_COLUMN] * 365) if current_year[COGS_COLUMN] > 0 else 0
    previous_dpo = (previous_year[ACCOUNTS_PAYABLE_COLUMN] / previous_year[COGS_COLUMN] * 365) if previous_year[COGS_COLUMN] > 0 else 0
    dpo_change = calculate_percentage_change(current_dpo, previous_dpo)

    current_dio = (current_year[INVENTORY_COLUMN] / current_year[COGS_COLUMN] * 365) if current_year[COGS_COLUMN] > 0 else 0
    previous_dio = (previous_year[INVENTORY_COLUMN] / previous_year[COGS_COLUMN] * 365) if previous_year[COGS_COLUMN] > 0 else 0
    dio_change = calculate_percentage_change(current_dio, previous_dio)

    # Return the metrics in a dictionary
    return {
        'current_sales': current_sales,
        'previous_sales': previous_sales,
        'sales_change': sales_change,
        'current_profit': current_profit,
        'previous_profit': previous_profit,
        'profit_change': profit_change,
        'current_dso': current_dso,
        'previous_dso': previous_dso,
        'dso_change': dso_change,
        'current_dpo': current_dpo,
        'previous_dpo': previous_dpo,
        'dpo_change': dpo_change,
        'current_dio': current_dio,
        'previous_dio': previous_dio,
        'dio_change': dio_change
    }

# Helper Functions
def add_sales_column(df):
    """
    Calculate the 'Sales' column based on 'Quantity' and 'UnitPrice' if not present.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: DataFrame with the 'Sales' column added if possible.
    """
    if SALES_COLUMN not in df.columns:
        if QUANTITY_COLUMN in df.columns and UNIT_PRICE_COLUMN in df.columns:
            # Calculate Sales as Quantity * UnitPrice
            df[SALES_COLUMN] = df[QUANTITY_COLUMN] * df[UNIT_PRICE_COLUMN]
            print("Sales column created from Quantity and UnitPrice.")
        else:
            print("Sales column not created: Quantity and/or UnitPrice column is missing.")
    return df

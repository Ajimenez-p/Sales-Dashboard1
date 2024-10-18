import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Read CSV as data frame, then check for missing vals and edit InvoiceNo for data type check
"""
df = pd.read_csv('sales_data.csv', parse_dates=['InvoiceDate'])
print(df.head())
print('\n')

# Check for missing values
print('Missing values:')
print(df.isnull().sum())
print('\n')
# Then replace:
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        # For numerical columns, fill with the mean
        df[column] = df[column].fillna(df[column].mean())
    else:
        # For non-numerical columns, fill with the mode
        df[column] = df[column].fillna(df[column].mode()[0])

# Remove any letters at the start of the "InvoiceNo" column
df['InvoiceNo'] = df['InvoiceNo'].str.replace('^[A-Za-z]+', '', regex=True)

"""
Convert the data types:
we need to do this for InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
UnitPrice, CustomerID, and Country
"""
for column in df.columns:
    try:
        match column:
            case 'InvoiceNo' | 'Quantity' | 'UnitPrice' | 'CustomerID':
                # Attempt to convert the column to int
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].astype(int)
                else:
                    raise ValueError(f'Data in column "{column}" unable to convert to int')
            case 'StockCode' | 'Description' | 'Country':
                # Attempt to convert the column to str
                if pd.api.types.is_string_dtype(df[column]):
                    df[column] = df[column].astype(str)
                else:
                    raise ValueError(f'Data in column "{column}" unable to convert to str')
            case 'InvoiceDate':
                # Attempt to convert the column to datetime
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column])
                else:
                    raise ValueError(f'Data in column "{column}" unable to convert to datetime')
            case _:
                raise TypeError(f'No conversion rule specified for column "{column}".')
    except (ValueError, TypeError) as e:
        # Handle the exception (e.g., log the error message)
        print(f"Error: {e}")
        continue

# Now with the data prepped, we can start using it
if 'Quantity' in df.columns and 'UnitPrice' in df.columns: 
    df['Sales'] = df['Quantity'] * df['UnitPrice']

# Describe our data
print(df.describe())
total_quantity = df['Quantity'].sum()
total_sales = df['Sales'].sum()
sales_by_country = df.groupby('Country')['Sales'].sum()
monthly_sales = df.resample('M', on='InvoiceDate')['Sales'].sum()
print(f'Total Quantity: {total_quantity}\n\nTotal Sales: {total_sales}')
print(f'Sales by Country: {sales_by_country}\n\nMonthly Sales: {monthly_sales}')

# Visualize our data, starting with plot of Sales by Country
plt.figure(figsize=(8, 6))
sns.barplot(x=sales_by_country.index, y=sales_by_country.values)
plt.title('Sales by Country')
plt.xlabel('Country')
plt.ylabel('Sales')
plt.show()

# Plot of monthly sales
plt.figure(figsize=(10, 6))
monthly_sales.plot()
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

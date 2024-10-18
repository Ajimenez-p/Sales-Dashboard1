# Imports
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
# Configurations and Constants

# Default coloring schemes
SALES_COLOR = '#1f77b4'
PROFIT_COLOR = '#2ca02c'
TARGET_COLOR = '#ff7f0e'

# Default chart settings
DEFAULT_TITLE_FONT_SIZE = 18
DEFAULT_AXIS_TITLE_FONT_SIZE = 14

# Visualization Functions
def create_sales_by_product_chart(df):
    """
    Creates a bar chart for sales by product.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart of sales by product.
    """
    fig = px.bar(
        df.groupby('Product')['Sales'].sum().reset_index(),
        x='Product',
        y='Sales',
        title='Sales by Product',
        color='Product',
        labels={'Sales': 'Total Sales', 'Product': 'Product'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        title={'text': 'Sales by Product', 'x': 0.5, 'xanchor': 'center'},
        title_font_size=DEFAULT_TITLE_FONT_SIZE
    )
    return fig

def create_sales_vs_target_chart(df):
    """
    Create a bar chart comparing actual sales vs. target.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data, including 'Sales' and 'SalesTarget' columns.

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart of actual sales vs. target.
    """

    # Resample data by month, aggregating 'Sales' and averaging 'SalesTarget'
    monthly_data = df.resample('ME', on='InvoiceDate').agg({
        'Sales': 'sum',
        'SalesTarget': 'mean'
    }).reset_index()

    # Extract the sales target values after resampling
    sales_target = monthly_data['SalesTarget']

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_data['InvoiceDate'],
        y=monthly_data['Sales'],
        name='Actual Sales',
        marker_color=SALES_COLOR
    ))
    fig.add_trace(go.Scatter(
        x=monthly_data['InvoiceDate'],
        y=sales_target,
        name='Sales Target',
        line=dict(color=TARGET_COLOR, dash='dash')
    ))

    # Update the layout
    fig.update_layout(
        title='Actual Sales vs. Target',
        xaxis_title='Month',
        yaxis_title='Sales',
        title_font_size=DEFAULT_TITLE_FONT_SIZE,
        xaxis_title_font_size=DEFAULT_AXIS_TITLE_FONT_SIZE,
        yaxis_title_font_size=DEFAULT_AXIS_TITLE_FONT_SIZE
    )
    return fig

def create_sales_by_region_chart(df):
    """
    Create a bar chart for sales by region.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart of sales by region.
    """
    fig = px.bar(
        df.groupby('Country')['Sales'].sum().reset_index(),
        x='Country',
        y='Sales',
        title='Sales by Region',
        color='Country',
        labels={'Sales': 'Total Sales', 'Country': 'Region'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        title={'text': 'Sales by Region', 'x': 0.5, 'xanchor': 'center'},
        title_font_size=DEFAULT_TITLE_FONT_SIZE
    )
    return fig

def create_monthly_sales_trend_chart(df):
    """
    Create a line chart showing the monthly sales trend.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        plotly.graph_objs._figure.Figure: Line chart of monthly sales.
    """
    monthly_sales = df.resample('ME', on='InvoiceDate')['Sales'].sum().reset_index()

    fig = px.line(
        monthly_sales,
        x='InvoiceDate',
        y='Sales',
        title='Monthly Sales Trend',
        labels={'Sales': 'Total Sales', 'InvoiceDate': 'Month'},
        line_shape='linear'
    )
    fig.update_layout(
        title={'text': 'Monthly Sales Trend', 'x': 0.5, 'xanchor': 'center'},
        title_font_size=DEFAULT_TITLE_FONT_SIZE
    )
    return fig

def create_sales_by_channel_pie_chart(df):
    """
    Create a pie chart showing sales distribution by channel.

    Args:
        df (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        plotly.graph_objs._figure.Figure: Pie chart of sales by channel.
    """
    sales_by_channel = df.groupby('Channel')['Sales'].sum().reset_index()

    fig = px.pie(
        sales_by_channel,
        names='Channel',
        values='Sales',
        title='Sales by Channel',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title={'text': 'Sales by Channel', 'x': 0.5, 'xanchor': 'center'},
        title_font_size=DEFAULT_TITLE_FONT_SIZE
    )
    return fig

# Helper Functions
def format_axis_titles(fig, x_title, y_title):
    """
    Format the axis titles for a Plotly figure.

    Args:
        fig (plotly.graph_objs._figure.Figure): The Plotly figure.
        x_title (str): Title for the x-axis.
        y_title (str): Title for the y-axis.
    """
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_title_font_size=DEFAULT_AXIS_TITLE_FONT_SIZE,
        yaxis_title_font_size=DEFAULT_AXIS_TITLE_FONT_SIZE
    )

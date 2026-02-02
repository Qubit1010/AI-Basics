"""
SALES DATA ANALYSIS DASHBOARD
==============================
A practical Pandas project analyzing sales data for a retail company

Features:
- Generate realistic sales data
- Sales performance analysis
- Product analysis
- Customer segmentation
- Time-based trends
- Regional analysis
- Generate comprehensive reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("SALES DATA ANALYSIS DASHBOARD")
print("=" * 80)

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
print("\n" + "=" * 80)
print("1. GENERATING SAMPLE SALES DATA")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range (last 12 months)
end_date = datetime(2024, 12, 31)
start_date = end_date - timedelta(days=365)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Product catalog
products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 
            'Webcam', 'USB Cable', 'SSD Drive', 'RAM', 'Graphics Card']
product_prices = {
    'Laptop': 899, 'Mouse': 25, 'Keyboard': 75, 'Monitor': 299,
    'Headphones': 149, 'Webcam': 89, 'USB Cable': 12, 'SSD Drive': 129,
    'RAM': 85, 'Graphics Card': 449
}

# Regions
regions = ['North', 'South', 'East', 'West']

# Sales channels
channels = ['Online', 'Retail Store', 'Phone']

# Generate sales data
num_transactions = 1000

data = {
    'TransactionID': [f'TXN{str(i).zfill(5)}' for i in range(1, num_transactions + 1)],
    'Date': np.random.choice(date_range, num_transactions),
    'Product': np.random.choice(products, num_transactions),
    'Quantity': np.random.randint(1, 6, num_transactions),
    'Region': np.random.choice(regions, num_transactions),
    'Channel': np.random.choice(channels, num_transactions),
    'CustomerID': [f'CUST{str(np.random.randint(1, 201)).zfill(4)}' for _ in range(num_transactions)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add Price and Revenue columns
df['Price'] = df['Product'].map(product_prices)
df['Revenue'] = df['Price'] * df['Quantity']

# Add discount (random 0-20%)
df['Discount_Percent'] = np.random.choice([0, 5, 10, 15, 20], num_transactions, p=[0.5, 0.2, 0.15, 0.1, 0.05])
df['Discount_Amount'] = df['Revenue'] * (df['Discount_Percent'] / 100)
df['Final_Revenue'] = df['Revenue'] - df['Discount_Amount']

# Add Month, Quarter, Year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.day_name()

print(f"\nGenerated {len(df)} sales transactions")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print("\nFirst 5 transactions:")
print(df[['TransactionID', 'Date', 'Product', 'Quantity', 'Final_Revenue', 'Region']].head())

# Save raw data
df.to_csv('/home/claude/sales_data.csv', index=False)
print(f"\nRaw data saved to: sales_data.csv")


# ============================================================================
# 2. OVERALL BUSINESS METRICS
# ============================================================================
print("\n" + "=" * 80)
print("2. OVERALL BUSINESS METRICS")
print("=" * 80)

total_revenue = df['Final_Revenue'].sum()
total_transactions = len(df)
total_quantity_sold = df['Quantity'].sum()
average_transaction_value = df['Final_Revenue'].mean()
total_customers = df['CustomerID'].nunique()

print(f"\n📊 Key Performance Indicators:")
print(f"  Total Revenue:              ${total_revenue:,.2f}")
print(f"  Total Transactions:         {total_transactions:,}")
print(f"  Total Items Sold:           {total_quantity_sold:,}")
print(f"  Average Transaction Value:  ${average_transaction_value:.2f}")
print(f"  Unique Customers:           {total_customers}")
print(f"  Average Revenue per Customer: ${total_revenue/total_customers:.2f}")


# ============================================================================
# 3. PRODUCT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. PRODUCT ANALYSIS")
print("=" * 80)

# Sales by product
product_analysis = df.groupby('Product').agg({
    'TransactionID': 'count',
    'Quantity': 'sum',
    'Final_Revenue': 'sum'
}).rename(columns={
    'TransactionID': 'Num_Transactions',
    'Quantity': 'Total_Quantity',
    'Final_Revenue': 'Total_Revenue'
})

product_analysis['Avg_Revenue_per_Transaction'] = (
    product_analysis['Total_Revenue'] / product_analysis['Num_Transactions']
)

product_analysis = product_analysis.sort_values('Total_Revenue', ascending=False)

print("\nProduct Performance:")
print(product_analysis)

# Top 5 products by revenue
print("\n🏆 Top 5 Products by Revenue:")
top_5_products = product_analysis.head(5)
for idx, (product, row) in enumerate(top_5_products.iterrows(), 1):
    print(f"  {idx}. {product:<15} - ${row['Total_Revenue']:>10,.2f} ({row['Num_Transactions']} transactions)")

# Bottom 3 products
print("\n⚠️  Bottom 3 Products by Revenue:")
bottom_3_products = product_analysis.tail(3)
for idx, (product, row) in enumerate(bottom_3_products.iterrows(), 1):
    print(f"  {idx}. {product:<15} - ${row['Total_Revenue']:>10,.2f}")


# ============================================================================
# 4. REGIONAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. REGIONAL ANALYSIS")
print("=" * 80)

regional_analysis = df.groupby('Region').agg({
    'TransactionID': 'count',
    'Final_Revenue': ['sum', 'mean'],
    'CustomerID': 'nunique'
}).round(2)

regional_analysis.columns = ['Transactions', 'Total_Revenue', 'Avg_Transaction', 'Unique_Customers']
regional_analysis = regional_analysis.sort_values('Total_Revenue', ascending=False)

print("\nRegional Performance:")
print(regional_analysis)

# Market share by region
regional_analysis['Market_Share_%'] = (
    regional_analysis['Total_Revenue'] / regional_analysis['Total_Revenue'].sum() * 100
).round(2)

print("\n📍 Market Share by Region:")
for region, row in regional_analysis.iterrows():
    bar = '█' * int(row['Market_Share_%'] / 2)
    print(f"  {region:<10} {row['Market_Share_%']:>5.1f}% {bar}")


# ============================================================================
# 5. CHANNEL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. SALES CHANNEL ANALYSIS")
print("=" * 80)

channel_analysis = df.groupby('Channel').agg({
    'TransactionID': 'count',
    'Final_Revenue': 'sum',
    'Quantity': 'sum'
}).rename(columns={
    'TransactionID': 'Transactions',
    'Final_Revenue': 'Revenue',
    'Quantity': 'Items_Sold'
})

channel_analysis['Revenue_%'] = (
    channel_analysis['Revenue'] / channel_analysis['Revenue'].sum() * 100
).round(2)

print("\nChannel Performance:")
print(channel_analysis.sort_values('Revenue', ascending=False))


# ============================================================================
# 6. TIME-BASED ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. TIME-BASED ANALYSIS")
print("=" * 80)

# Monthly trends
monthly_sales = df.groupby('Month_Name').agg({
    'Final_Revenue': 'sum',
    'TransactionID': 'count'
}).rename(columns={
    'Final_Revenue': 'Revenue',
    'TransactionID': 'Transactions'
})

# Reorder by month
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales = monthly_sales.reindex([m for m in month_order if m in monthly_sales.index])

print("\n📅 Monthly Sales Trends:")
print(monthly_sales)

# Find best and worst months
best_month = monthly_sales['Revenue'].idxmax()
worst_month = monthly_sales['Revenue'].idxmin()
print(f"\nBest Month:  {best_month} (${monthly_sales.loc[best_month, 'Revenue']:,.2f})")
print(f"Worst Month: {worst_month} (${monthly_sales.loc[worst_month, 'Revenue']:,.2f})")

# Quarterly analysis
quarterly_sales = df.groupby('Quarter')['Final_Revenue'].sum()
print("\n📊 Quarterly Sales:")
for quarter, revenue in quarterly_sales.items():
    print(f"  Q{quarter}: ${revenue:,.2f}")

# Day of week analysis
dow_sales = df.groupby('DayOfWeek')['Final_Revenue'].sum().sort_values(ascending=False)
print("\n📆 Sales by Day of Week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sales = dow_sales.reindex([d for d in day_order if d in dow_sales.index])
for day, revenue in dow_sales.items():
    print(f"  {day:<10} ${revenue:,.2f}")


# ============================================================================
# 7. CUSTOMER ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. CUSTOMER ANALYSIS")
print("=" * 80)

# Customer purchase behavior
customer_analysis = df.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Final_Revenue': 'sum',
    'Quantity': 'sum'
}).rename(columns={
    'TransactionID': 'Num_Purchases',
    'Final_Revenue': 'Total_Spent',
    'Quantity': 'Items_Purchased'
})

customer_analysis['Avg_Purchase_Value'] = (
    customer_analysis['Total_Spent'] / customer_analysis['Num_Purchases']
).round(2)

# Customer segmentation
customer_analysis['Segment'] = pd.cut(
    customer_analysis['Total_Spent'],
    bins=[0, 500, 2000, float('inf')],
    labels=['Bronze', 'Silver', 'Gold']
)

print("\n👥 Customer Segmentation:")
segment_counts = customer_analysis['Segment'].value_counts()
for segment, count in segment_counts.items():
    percentage = (count / len(customer_analysis) * 100)
    print(f"  {segment:<8} {count:>4} customers ({percentage:.1f}%)")

# Top 10 customers
top_customers = customer_analysis.nlargest(10, 'Total_Spent')
print("\n🌟 Top 10 Customers by Revenue:")
for idx, (cust_id, row) in enumerate(top_customers.iterrows(), 1):
    print(f"  {idx:2}. {cust_id} - ${row['Total_Spent']:>8,.2f} ({row['Num_Purchases']} purchases) - {row['Segment']}")

# Calculate customer lifetime metrics
print("\n📈 Customer Metrics:")
print(f"  Average purchases per customer:    {customer_analysis['Num_Purchases'].mean():.2f}")
print(f"  Average spend per customer:        ${customer_analysis['Total_Spent'].mean():.2f}")
print(f"  Median spend per customer:         ${customer_analysis['Total_Spent'].median():.2f}")


# ============================================================================
# 8. DISCOUNT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("8. DISCOUNT ANALYSIS")
print("=" * 80)

discount_analysis = df.groupby('Discount_Percent').agg({
    'TransactionID': 'count',
    'Discount_Amount': 'sum',
    'Final_Revenue': 'sum'
}).rename(columns={
    'TransactionID': 'Transactions',
    'Discount_Amount': 'Total_Discount_Given',
    'Final_Revenue': 'Revenue_After_Discount'
})

print("\n💰 Discount Impact:")
print(discount_analysis)

total_discount_given = df['Discount_Amount'].sum()
revenue_without_discount = df['Revenue'].sum()
print(f"\nTotal Discount Given: ${total_discount_given:,.2f}")
print(f"Discount as % of Revenue: {(total_discount_given/revenue_without_discount*100):.2f}%")


# ============================================================================
# 9. CROSS-SELLING ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. CROSS-SELLING OPPORTUNITIES")
print("=" * 80)

# Products frequently bought together
customer_products = df.groupby('CustomerID')['Product'].apply(list).reset_index()
customer_products['Num_Different_Products'] = customer_products['Product'].apply(lambda x: len(set(x)))

# Find customers who bought multiple products
multi_product_customers = customer_products[customer_products['Num_Different_Products'] > 1]

print(f"\n🛒 Cross-selling Insights:")
print(f"  Customers who bought 1 product only:  {len(customer_products[customer_products['Num_Different_Products'] == 1])}")
print(f"  Customers who bought multiple products: {len(multi_product_customers)}")
print(f"  Cross-selling rate: {(len(multi_product_customers)/len(customer_products)*100):.1f}%")

# Most common product combinations
print("\n🔗 Product Combination Analysis:")
product_pairs = df.groupby(['CustomerID', 'Product']).size().reset_index(name='count')
customers_with_multiple = product_pairs.groupby('CustomerID').size()
customers_with_multiple = customers_with_multiple[customers_with_multiple > 1].index

if len(customers_with_multiple) > 0:
    sample_combos = df[df['CustomerID'].isin(customers_with_multiple.tolist()[:5])]
    combo_summary = sample_combos.groupby('CustomerID')['Product'].apply(lambda x: ', '.join(set(x)))
    print("Sample customer product combinations:")
    for cust_id, products in combo_summary.head(5).items():
        print(f"  {cust_id}: {products}")


# ============================================================================
# 10. REVENUE FORECASTING (Simple Moving Average)
# ============================================================================
print("\n" + "=" * 80)
print("10. REVENUE TRENDS & FORECASTING")
print("=" * 80)

# Daily revenue
daily_revenue = df.groupby('Date')['Final_Revenue'].sum().sort_index()

# Calculate 7-day moving average
daily_revenue_df = daily_revenue.reset_index()
daily_revenue_df['7_Day_MA'] = daily_revenue_df['Final_Revenue'].rolling(window=7).mean()

print("\nDaily Revenue Statistics:")
print(f"  Average daily revenue:  ${daily_revenue.mean():,.2f}")
print(f"  Best day:               ${daily_revenue.max():,.2f} on {daily_revenue.idxmax().date()}")
print(f"  Worst day:              ${daily_revenue.min():,.2f} on {daily_revenue.idxmin().date()}")

# Recent trend
recent_30_days = daily_revenue.tail(30).mean()
previous_30_days = daily_revenue.tail(60).head(30).mean()
trend = ((recent_30_days - previous_30_days) / previous_30_days * 100)

print(f"\n📈 Revenue Trend (Last 30 vs Previous 30 days):")
print(f"  Recent 30 days avg:     ${recent_30_days:,.2f}")
print(f"  Previous 30 days avg:   ${previous_30_days:,.2f}")
print(f"  Trend:                  {trend:+.2f}% {'📈' if trend > 0 else '📉'}")


# ============================================================================
# 11. PERFORMANCE SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("11. EXECUTIVE SUMMARY REPORT")
print("=" * 80)

report = f"""
SALES PERFORMANCE EXECUTIVE SUMMARY
{'='*80}

PERIOD: {df['Date'].min().date()} to {df['Date'].max().date()}

FINANCIAL METRICS:
  • Total Revenue:                ${total_revenue:,.2f}
  • Average Transaction Value:    ${average_transaction_value:.2f}
  • Total Discounts Given:        ${total_discount_given:,.2f}

SALES METRICS:
  • Total Transactions:           {total_transactions:,}
  • Total Items Sold:             {total_quantity_sold:,}
  • Average Items per Transaction: {total_quantity_sold/total_transactions:.2f}

CUSTOMER METRICS:
  • Total Customers:              {total_customers}
  • Average Revenue per Customer: ${total_revenue/total_customers:.2f}
  • Cross-selling Rate:           {(len(multi_product_customers)/len(customer_products)*100):.1f}%

TOP PERFORMERS:
  • Best Product:                 {product_analysis.index[0]} (${product_analysis.iloc[0]['Total_Revenue']:,.2f})
  • Best Region:                  {regional_analysis.index[0]} (${regional_analysis.iloc[0]['Total_Revenue']:,.2f})
  • Best Channel:                 {channel_analysis['Revenue'].idxmax()} (${channel_analysis['Revenue'].max():,.2f})
  • Best Month:                   {best_month} (${monthly_sales.loc[best_month, 'Revenue']:,.2f})

AREAS FOR IMPROVEMENT:
  • Lowest Product:               {product_analysis.index[-1]} (${product_analysis.iloc[-1]['Total_Revenue']:,.2f})
  • Lowest Region:                {regional_analysis.index[-1]} (${regional_analysis.iloc[-1]['Total_Revenue']:,.2f})
  • Worst Month:                  {worst_month} (${monthly_sales.loc[worst_month, 'Revenue']:,.2f})

TRENDS:
  • 30-Day Trend:                 {trend:+.2f}%
  • Customer Segments:            Gold: {segment_counts.get('Gold', 0)}, Silver: {segment_counts.get('Silver', 0)}, Bronze: {segment_counts.get('Bronze', 0)}

{'='*80}
"""

print(report)

# Save report to file
with open('/home/claude/executive_summary.txt', 'w') as f:
    f.write(report)

print("Executive summary saved to: executive_summary.txt")


# ============================================================================
# 12. EXPORT DETAILED REPORTS
# ============================================================================
print("\n" + "=" * 80)
print("12. EXPORTING DETAILED REPORTS")
print("=" * 80)

# Export to Excel with multiple sheets
excel_file = '/home/claude/sales_analysis_report.xlsx'

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # Sheet 1: Raw Data
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    # Sheet 2: Product Analysis
    product_analysis.to_excel(writer, sheet_name='Product_Analysis')
    
    # Sheet 3: Regional Analysis
    regional_analysis.to_excel(writer, sheet_name='Regional_Analysis')
    
    # Sheet 4: Monthly Trends
    monthly_sales.to_excel(writer, sheet_name='Monthly_Trends')
    
    # Sheet 5: Customer Analysis
    customer_analysis.to_excel(writer, sheet_name='Customer_Analysis')
    
    # Sheet 6: Channel Analysis
    channel_analysis.to_excel(writer, sheet_name='Channel_Analysis')

print(f"Detailed Excel report saved to: {excel_file}")

# Export CSV reports
product_analysis.to_csv('/home/claude/product_analysis.csv')
regional_analysis.to_csv('/home/claude/regional_analysis.csv')
customer_analysis.to_csv('/home/claude/customer_analysis.csv')

print("\nCSV reports saved:")
print("  - product_analysis.csv")
print("  - regional_analysis.csv")
print("  - customer_analysis.csv")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Key Insights Generated:")
insights = [
    f"✓ Analyzed {total_transactions:,} transactions from {total_customers} customers",
    f"✓ Total revenue: ${total_revenue:,.2f}",
    f"✓ Best performing product: {product_analysis.index[0]}",
    f"✓ Best performing region: {regional_analysis.index[0]}",
    f"✓ Revenue trend: {trend:+.2f}% (last 30 days)",
    f"✓ Cross-selling rate: {(len(multi_product_customers)/len(customer_products)*100):.1f}%"
]

for insight in insights:
    print(f"  {insight}")

print("\n📁 Files Generated:")
files = [
    "sales_data.csv - Raw transaction data",
    "executive_summary.txt - Executive summary report",
    "sales_analysis_report.xlsx - Comprehensive Excel report (6 sheets)",
    "product_analysis.csv - Product performance metrics",
    "regional_analysis.csv - Regional performance metrics",
    "customer_analysis.csv - Customer segmentation data"
]

for file in files:
    print(f"  • {file}")

print("\n" + "=" * 80)

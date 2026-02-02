import pandas as pd
import numpy as np

print("Pandas version:", pd.__version__)
print("=" * 80)

# ============================================================================
# 1. PANDAS SERIES (1D Data Structure)
# ============================================================================
print("\n" + "=" * 80)
print("1. PANDAS SERIES (1D labeled array)")
print("=" * 80)

# 1.1 Creating Series from list
print("\n--- 1.1 Creating Series from List ---")
data_list = [10, 20, 30, 40, 50]
series1 = pd.Series(data_list)
print("Series from list:")
print(series1)
print(f"\nType: {type(series1)}")

# 1.2 Series with custom index
print("\n--- 1.2 Series with Custom Index ---")
series2 = pd.Series([100, 200, 300], index=['a', 'b', 'c'])
print(series2)

# 1.3 Series from dictionary
print("\n--- 1.3 Series from Dictionary ---")
data_dict = {'apple': 5, 'banana': 10, 'orange': 15}
series3 = pd.Series(data_dict)
print(series3)

# 1.4 Series attributes
print("\n--- 1.4 Series Attributes ---")
print(f"Values: {series2.values}")
print(f"Index: {series2.index}")
print(f"Shape: {series2.shape}")
print(f"Size: {series2.size}")
print(f"Data type: {series2.dtype}")

# 1.5 Accessing elements
print("\n--- 1.5 Accessing Elements ---")
print(f"Element at index 'b': {series2['b']}")
# Updated to use .iloc for position-based access
print(f"Element at position 0: {series2.iloc[0]}")
print(f"Slice [0:2]:\n{series2.iloc[0:2]}")

# ============================================================================
# 2. PANDAS DATAFRAME (2D Data Structure)
# ============================================================================
print("\n" + "=" * 80)
print("2. PANDAS DATAFRAME (2D labeled data structure)")
print("=" * 80)

# 2.1 Creating DataFrame from dictionary
print("\n--- 2.1 Creating DataFrame from Dictionary ---")
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 55000, 52000, 58000]
}
df = pd.DataFrame(data)
print(df)

# 2.2 Creating DataFrame from list of lists
print("\n--- 2.2 Creating DataFrame from List of Lists ---")
data_list = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'London'],
    ['Charlie', 35, 'Paris']
]
df2 = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])
print(df2)

# 2.3 Creating DataFrame from NumPy array
print("\n--- 2.3 Creating DataFrame from NumPy Array ---")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df3)

# ============================================================================
# 3. DATAFRAME ATTRIBUTES AND METHODS
# ============================================================================
print("\n" + "=" * 80)
print("3. DATAFRAME ATTRIBUTES AND METHODS")
print("=" * 80)

print("\n--- Basic Information ---")
print(f"Shape (rows, columns): {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")

print("\n--- First and Last Rows ---")
print("First 3 rows (head):")
print(df.head(3))

print("\nLast 2 rows (tail):")
print(df.tail(2))

print("\n--- Info and Description ---")
print("\ndf.info():")
df.info()

print("\ndf.describe() - Statistical summary:")
print(df.describe())

# ============================================================================
# 4. SELECTING DATA
# ============================================================================
print("\n" + "=" * 80)
print("4. SELECTING DATA")
print("=" * 80)

# 4.1 Selecting columns
print("\n--- 4.1 Selecting Columns ---")
print("Single column (returns Series):")
print(df['Name'])
print(f"\nType: {type(df['Name'])}")

print("\nMultiple columns (returns DataFrame):")
print(df[['Name', 'Age']])

# 4.2 Selecting rows by index position (iloc)
print("\n--- 4.2 Selecting Rows by Position (iloc) ---")
print("First row:")
print(df.iloc[0])

print("\nFirst 3 rows:")
print(df.iloc[0:3])

print("\nSpecific rows and columns:")
print(df.iloc[0:3, 0:2])  # First 3 rows, first 2 columns

# 4.3 Selecting rows by label (loc)
print("\n--- 4.3 Selecting Rows by Label (loc) ---")
print("Row with index 0:")
print(df.loc[0])

print("\nRows 0 to 2 with specific columns:")
print(df.loc[0:2, ['Name', 'City']])

# 4.4 Boolean indexing
print("\n--- 4.4 Boolean Indexing (Filtering) ---")
print("People older than 30:")
print(df[df['Age'] > 30])

print("\nPeople with Salary > 55000:")
print(df[df['Salary'] > 55000])

print("\nMultiple conditions (Age > 25 AND Salary > 52000):")
print(df[(df['Age'] > 25) & (df['Salary'] > 52000)])

print("\nPeople from New York OR Paris:")
print(df[(df['City'] == 'New York') | (df['City'] == 'Paris')])

# ============================================================================
# 5. ADDING AND MODIFYING DATA
# ============================================================================
print("\n" + "=" * 80)
print("5. ADDING AND MODIFYING DATA")
print("=" * 80)

# Create a copy to modify
df_copy = df.copy()

# 5.1 Adding new column
print("\n--- 5.1 Adding New Column ---")
df_copy['Department'] = ['HR', 'IT', 'Finance', 'Marketing', 'IT']
print(df_copy)

# 5.2 Adding calculated column
print("\n--- 5.2 Adding Calculated Column ---")
df_copy['Annual_Salary'] = df_copy['Salary'] * 12
print(df_copy[['Name', 'Salary', 'Annual_Salary']])

# 5.3 Modifying existing values
print("\n--- 5.3 Modifying Values ---")
df_copy.loc[0, 'Salary'] = 55000
print("After modifying Alice's salary:")
print(df_copy[['Name', 'Salary']])

# 5.4 Adding new row
print("\n--- 5.4 Adding New Row ---")
new_row = pd.DataFrame({
    'Name': ['Frank'],
    'Age': [27],
    'City': ['Berlin'],
    'Salary': [53000],
    'Department': ['Sales'],
    'Annual_Salary': [636000]
})
df_copy = pd.concat([df_copy, new_row], ignore_index=True)
print(df_copy)

# ============================================================================
# 6. DELETING DATA
# ============================================================================
print("\n" + "=" * 80)
print("6. DELETING DATA")
print("=" * 80)

df_delete = df.copy()

# 6.1 Dropping columns
print("\n--- 6.1 Dropping Columns ---")
df_dropped_col = df_delete.drop('City', axis=1)
print(df_dropped_col)

print("\nDropping multiple columns:")
df_dropped_cols = df_delete.drop(['City', 'Salary'], axis=1)
print(df_dropped_cols)

# 6.2 Dropping rows
print("\n--- 6.2 Dropping Rows ---")
df_dropped_row = df_delete.drop(0, axis=0)  # Drop row with index 0
print(df_dropped_row)

# 6.3 Dropping rows based on condition
print("\n--- 6.3 Dropping Based on Condition ---")
df_filtered = df_delete[df_delete['Age'] <= 30]
print("Keep only people aged 30 or younger:")
print(df_filtered)

# ============================================================================
# 7. HANDLING MISSING DATA
# ============================================================================
print("\n" + "=" * 80)
print("7. HANDLING MISSING DATA")
print("=" * 80)

# Create DataFrame with missing values
data_missing = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, 50],
    'C': [100, 200, 300, np.nan, 500]
}
df_missing = pd.DataFrame(data_missing)
print("DataFrame with missing values:")
print(df_missing)

# 7.1 Detecting missing values
print("\n--- 7.1 Detecting Missing Values ---")
print("isnull():")
print(df_missing.isnull())

print("\nSum of missing values per column:")
print(df_missing.isnull().sum())

# 7.2 Dropping missing values
print("\n--- 7.2 Dropping Missing Values ---")
print("Drop any row with missing values:")
print(df_missing.dropna())

print("\nDrop columns with any missing values:")
print(df_missing.dropna(axis=1))

# 7.3 Filling missing values
print("\n--- 7.3 Filling Missing Values ---")
print("Fill with 0:")
print(df_missing.fillna(0))

print("\nFill with mean of each column:")
print(df_missing.fillna(df_missing.mean()))

# Updated to use .ffill() method directly
print("\nForward fill (use previous value):")
print(df_missing.ffill())

# ============================================================================
# 8. SORTING DATA
# ============================================================================
print("\n" + "=" * 80)
print("8. SORTING DATA")
print("=" * 80)

# 8.1 Sort by values
print("\n--- 8.1 Sort by Values ---")
print("Sort by Age (ascending):")
print(df.sort_values('Age'))

print("\nSort by Salary (descending):")
print(df.sort_values('Salary', ascending=False))

print("\nSort by multiple columns:")
print(df.sort_values(['Age', 'Salary'], ascending=[True, False]))

# 8.2 Sort by index
print("\n--- 8.2 Sort by Index ---")
df_shuffled = df.sample(frac=1)  # Shuffle
print("Shuffled:")
print(df_shuffled)

print("\nSorted by index:")
print(df_shuffled.sort_index())

# ============================================================================
# 9. GROUPING AND AGGREGATION
# ============================================================================
print("\n" + "=" * 80)
print("9. GROUPING AND AGGREGATION")
print("=" * 80)

# Create sample data
data_sales = {
    'Region': ['East', 'West', 'East', 'West', 'East', 'West'],
    'Product': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 180, 120, 220],
    'Quantity': [10, 15, 20, 18, 12, 22]
}
df_sales = pd.DataFrame(data_sales)
print("Sales data:")
print(df_sales)

# 9.1 GroupBy single column
print("\n--- 9.1 GroupBy Single Column ---")
print("Average sales by Region:")
print(df_sales.groupby('Region')['Sales'].mean())

print("\nSum of sales by Product:")
print(df_sales.groupby('Product')['Sales'].sum())

# 9.2 Multiple aggregations
print("\n--- 9.2 Multiple Aggregations ---")
print("Multiple statistics by Region:")
print(df_sales.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']))

# 9.3 GroupBy multiple columns
print("\n--- 9.3 GroupBy Multiple Columns ---")
print("Group by Region and Product:")
print(df_sales.groupby(['Region', 'Product'])['Sales'].sum())

# ============================================================================
# 10. MERGING AND JOINING
# ============================================================================
print("\n" + "=" * 80)
print("10. MERGING AND JOINING")
print("=" * 80)

# Create sample DataFrames
df_employees = pd.DataFrame({
    'EmployeeID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'IT', 'Finance', 'IT']
})

df_salaries = pd.DataFrame({
    'EmployeeID': [1, 2, 3, 5],
    'Salary': [50000, 60000, 55000, 52000]
})

print("Employees:")
print(df_employees)
print("\nSalaries:")
print(df_salaries)

# 10.1 Inner join (only matching records)
print("\n--- 10.1 Inner Join ---")
merged_inner = pd.merge(df_employees, df_salaries, on='EmployeeID', how='inner')
print(merged_inner)

# 10.2 Left join (all from left, matching from right)
print("\n--- 10.2 Left Join ---")
merged_left = pd.merge(df_employees, df_salaries, on='EmployeeID', how='left')
print(merged_left)

# 10.3 Right join
print("\n--- 10.3 Right Join ---")
merged_right = pd.merge(df_employees, df_salaries, on='EmployeeID', how='right')
print(merged_right)

# 10.4 Outer join (all records from both)
print("\n--- 10.4 Outer Join ---")
merged_outer = pd.merge(df_employees, df_salaries, on='EmployeeID', how='outer')
print(merged_outer)

# 10.5 Concatenating DataFrames
print("\n--- 10.5 Concatenation ---")
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

print("Vertical concatenation (rows):")
print(pd.concat([df1, df2], ignore_index=True))

print("\nHorizontal concatenation (columns):")
print(pd.concat([df1, df2], axis=1))

# ============================================================================
# 11. APPLYING FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("11. APPLYING FUNCTIONS")
print("=" * 80)

df_func = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})
print("Original DataFrame:")
print(df_func)

# 11.1 Apply to column
print("\n--- 11.1 Apply to Column ---")
df_func['A_squared'] = df_func['A'].apply(lambda x: x ** 2)
print(df_func)

# 11.2 Apply to multiple columns
print("\n--- 11.2 Apply to Multiple Columns ---")
df_func['Sum'] = df_func.apply(lambda row: row['A'] + row['B'], axis=1)
print(df_func)

# 11.3 Map values
print("\n--- 11.3 Map Values ---")
mapping = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}
df_func['A_text'] = df_func['A'].map(mapping)
print(df_func)

# ============================================================================
# 12. STRING OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("12. STRING OPERATIONS")
print("=" * 80)

df_str = pd.DataFrame({
    'Name': ['  alice  ', 'BOB', 'charlie', 'DAVID'],
    'Email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'david@email.com']
})
print("Original:")
print(df_str)

print("\n--- String Methods ---")
df_str['Name_clean'] = df_str['Name'].str.strip().str.title()
df_str['Email_domain'] = df_str['Email'].str.split('@').str[1]
df_str['Name_length'] = df_str['Name'].str.len()
print(df_str)

# ============================================================================
# 13. DATETIME OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("13. DATETIME OPERATIONS")
print("=" * 80)

# Create DataFrame with dates
df_dates = pd.DataFrame({
    'Date': ['2024-01-01', '2024-02-15', '2024-03-30', '2024-06-20'],
    'Sales': [100, 150, 200, 180]
})

print("Original:")
print(df_dates)

# Convert to datetime
df_dates['Date'] = pd.to_datetime(df_dates['Date'])
print(f"\nDate column type: {df_dates['Date'].dtype}")

# Extract date components
df_dates['Year'] = df_dates['Date'].dt.year
df_dates['Month'] = df_dates['Date'].dt.month
df_dates['Day'] = df_dates['Date'].dt.day
df_dates['DayOfWeek'] = df_dates['Date'].dt.day_name()

print("\nWith extracted components:")
print(df_dates)

# ============================================================================
# 14. READING AND WRITING FILES
# ============================================================================
print("\n" + "=" * 80)
print("14. READING AND WRITING FILES")
print("=" * 80)

# Read from CSV
df_read = pd.read_csv("data_pandas.csv")
print("\nData read from CSV:")
print(df_read)

# ============================================================================
# 15. PIVOT TABLES
# ============================================================================
print("\n" + "=" * 80)
print("15. PIVOT TABLES")
print("=" * 80)

# Create sample data
df_pivot = pd.DataFrame({
    'Date': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-01', '2024-02'],
    'Region': ['East', 'West', 'East', 'West', 'East', 'West'],
    'Product': ['A', 'A', 'B', 'B', 'B', 'A'],
    'Sales': [100, 150, 200, 180, 120, 220]
})

print("Original data:")
print(df_pivot)

print("\n--- Pivot Table ---")
pivot = df_pivot.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)
print(pivot)

# ============================================================================
# 16. USEFUL OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("16. USEFUL OPERATIONS")
print("=" * 80)

df_ops = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['x', 'y', 'x', 'z', 'y']
})

print("Original DataFrame:")
print(df_ops)

# 16.1 Unique values
print("\n--- Unique Values ---")
print(f"Unique values in C: {df_ops['C'].unique()}")
print(f"Number of unique values: {df_ops['C'].nunique()}")

# 16.2 Value counts
print("\n--- Value Counts ---")
print(df_ops['C'].value_counts())

# 16.3 Replace values
print("\n--- Replace Values ---")
df_ops_replaced = df_ops.replace({'x': 'X', 'y': 'Y', 'z': 'Z'})
print(df_ops_replaced)

# 16.4 Rename columns
print("\n--- Rename Columns ---")
df_renamed = df_ops.rename(columns={'A': 'Column_A', 'B': 'Column_B', 'C': 'Column_C'})
print(df_renamed)

# 16.5 Reset index
print("\n--- Reset Index ---")
df_reset = df_ops.set_index('C')
print("After setting C as index:")
print(df_reset)
print("\nAfter reset_index():")
print(df_reset.reset_index())

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PANDAS BASICS COVERED:")
print("=" * 80)

topics = [
    "1. Series - 1D labeled arrays",
    "2. DataFrame - 2D labeled data structure",
    "3. DataFrame attributes and methods",
    "4. Selecting data (loc, iloc, boolean indexing)",
    "5. Adding and modifying data",
    "6. Deleting data (drop)",
    "7. Handling missing data (dropna, fillna)",
    "8. Sorting data (sort_values, sort_index)",
    "9. Grouping and aggregation (groupby, agg)",
    "10. Merging and joining (merge, concat)",
    "11. Applying functions (apply, map)",
    "12. String operations (str methods)",
    "13. DateTime operations",
    "14. Reading/Writing files (CSV, Excel)",
    "15. Pivot tables",
    "16. Useful operations (unique, value_counts, rename, etc.)"
]

for topic in topics:
    print(f"✓ {topic}")

print("\n" + "=" * 80)
print("All Pandas basics covered successfully!")
print("=" * 80)
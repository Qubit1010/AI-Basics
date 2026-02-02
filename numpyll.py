"""
NUMPY BASICS - COMPLETE BEGINNER'S GUIDE
=========================================
Starting with Array Creation and covering all fundamental concepts
"""

import numpy as np

print("NumPy version:", np.__version__)
print("=" * 70)

# ============================================================================
# 1. CREATING ARRAYS - Different Methods
# ============================================================================
print("\n" + "=" * 70)
print("1. CREATING ARRAYS")
print("=" * 70)

# 1.1 From Python Lists
print("\n--- 1.1 Creating from Python Lists ---")
list1 = [1, 2, 3, 4, 5]
arr1 = np.array(list1)
print(f"From list: {arr1}")
print(f"Type: {type(arr1)}")

# 2D array from nested list
list2d = [[1, 2, 3], [4, 5, 6]]
arr2d = np.array(list2d)
print(f"\n2D array:\n{arr2d}")

# 1.2 Using arange() - similar to Python's range()
print("\n--- 1.2 Using arange() ---")
arr_range = np.arange(10)  # 0 to 9
print(f"arange(10): {arr_range}")

arr_range2 = np.arange(5, 15)  # 5 to 14
print(f"arange(5, 15): {arr_range2}")

arr_range3 = np.arange(0, 20, 2)  # 0 to 20, step 2
print(f"arange(0, 20, 2): {arr_range3}")

arr_range4 = np.arange(1.0, 5.0, 0.5)  # Works with floats
print(f"arange(1.0, 5.0, 0.5): {arr_range4}")

# 1.3 Using linspace() - evenly spaced values
print("\n--- 1.3 Using linspace() ---")
arr_lin = np.linspace(0, 10, 5)  # 5 values from 0 to 10
print(f"linspace(0, 10, 5): {arr_lin}")

arr_lin2 = np.linspace(0, 1, 11)  # 11 values from 0 to 1
print(f"linspace(0, 1, 11): {arr_lin2}")

# 1.4 Arrays of Zeros, Ones, and Empty
print("\n--- 1.4 Zeros, Ones, and Empty ---")
zeros = np.zeros(5)
print(f"zeros(5): {zeros}")

zeros_2d = np.zeros((3, 4))  # 3 rows, 4 columns
print(f"zeros (3x4):\n{zeros_2d}")

ones = np.ones(5)
print(f"\nones(5): {ones}")

ones_2d = np.ones((2, 3))
print(f"ones (2x3):\n{ones_2d}")

empty = np.empty(5)  # Uninitialized (random values)
print(f"\nempty(5): {empty}")

# 1.5 Identity Matrix
print("\n--- 1.5 Identity Matrix ---")
identity = np.eye(4)  # 4x4 identity matrix
print(f"eye(4):\n{identity}")

# 1.6 Arrays with specific values
print("\n--- 1.6 Arrays with Specific Values ---")
full_array = np.full(5, 7)  # Array of 5 sevens
print(f"full(5, 7): {full_array}")

full_2d = np.full((3, 3), 3.14)
print(f"full((3, 3), 3.14):\n{full_2d}")

# 1.7 Random Arrays
print("\n--- 1.7 Random Arrays ---")
random_arr = np.random.rand(5)  # Random values between 0 and 1
print(f"random.rand(5): {random_arr}")

random_2d = np.random.rand(3, 3)
print(f"random.rand(3, 3):\n{random_2d}")

random_int = np.random.randint(0, 10, 5)  # Random integers
print(f"random.randint(0, 10, 5): {random_int}")

random_normal = np.random.randn(5)  # Normal distribution
print(f"random.randn(5): {random_normal}")


# ============================================================================
# 2. ARRAY ATTRIBUTES
# ============================================================================
print("\n" + "=" * 70)
print("2. ARRAY ATTRIBUTES")
print("=" * 70)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Array:\n{arr}")
print(f"\nShape (dimensions): {arr.shape}")  # (rows, columns)
print(f"Number of dimensions: {arr.ndim}")
print(f"Total elements: {arr.size}")
print(f"Data type: {arr.dtype}")
print(f"Item size (bytes): {arr.itemsize}")
print(f"Total bytes: {arr.nbytes}")


# ============================================================================
# 3. DATA TYPES
# ============================================================================
print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

# Different data types
int_arr = np.array([1, 2, 3], dtype=np.int32)
print(f"int32 array: {int_arr}, dtype: {int_arr.dtype}")

float_arr = np.array([1, 2, 3], dtype=np.float64)
print(f"float64 array: {float_arr}, dtype: {float_arr.dtype}")

bool_arr = np.array([True, False, True], dtype=np.bool_)
print(f"bool array: {bool_arr}, dtype: {bool_arr.dtype}")

# Type conversion
arr_int = np.array([1.5, 2.7, 3.9])
arr_converted = arr_int.astype(np.int32)
print(f"\nOriginal: {arr_int}")
print(f"Converted to int: {arr_converted}")


# ============================================================================
# 4. ARRAY INDEXING AND SLICING
# ============================================================================
print("\n" + "=" * 70)
print("4. INDEXING AND SLICING")
print("=" * 70)

# 1D Array
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print(f"Array: {arr_1d}")
print(f"First element arr[0]: {arr_1d[0]}")
print(f"Last element arr[-1]: {arr_1d[-1]}")
print(f"Slice arr[2:5]: {arr_1d[2:5]}")
print(f"Slice arr[:4]: {arr_1d[:4]}")
print(f"Slice arr[5:]: {arr_1d[5:]}")
print(f"Every 2nd element arr[::2]: {arr_1d[::2]}")
print(f"Reverse arr[::-1]: {arr_1d[::-1]}")

# 2D Array
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Element at [0, 0]: {arr_2d[0, 0]}")
print(f"Element at [1, 2]: {arr_2d[1, 2]}")
print(f"First row arr[0]: {arr_2d[0]}")
print(f"First column arr[:, 0]: {arr_2d[:, 0]}")
print(f"Last column arr[:, -1]: {arr_2d[:, -1]}")
print(f"Subarray arr[0:2, 1:3]:\n{arr_2d[0:2, 1:3]}")


# ============================================================================
# 5. ARRAY RESHAPING
# ============================================================================
print("\n" + "=" * 70)
print("5. RESHAPING ARRAYS")
print("=" * 70)

arr = np.arange(12)
print(f"Original 1D array: {arr}")

reshaped = arr.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{reshaped}")

reshaped2 = arr.reshape(4, 3)
print(f"\nReshaped to 4x3:\n{reshaped2}")

reshaped3 = arr.reshape(2, 2, 3)  # 3D array
print(f"\nReshaped to 2x2x3:\n{reshaped3}")

# Flatten back to 1D
flattened = reshaped.flatten()
print(f"\nFlattened: {flattened}")

# Transpose
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nOriginal:\n{arr_2d}")
print(f"Transposed:\n{arr_2d.T}")


# ============================================================================
# 6. ARRAY OPERATIONS (Element-wise)
# ============================================================================
print("\n" + "=" * 70)
print("6. ARRAY OPERATIONS")
print("=" * 70)

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")

# Arithmetic operations
print(f"\nAddition: {arr1 + arr2}")
print(f"Subtraction: {arr2 - arr1}")
print(f"Multiplication: {arr1 * arr2}")
print(f"Division: {arr2 / arr1}")
print(f"Power: {arr1 ** 2}")
print(f"Square root: {np.sqrt(arr1)}")

# Operations with scalars
print(f"\narr1 + 10: {arr1 + 10}")
print(f"arr1 * 2: {arr1 * 2}")
print(f"arr1 ** 3: {arr1 ** 3}")


# ============================================================================
# 7. MATHEMATICAL FUNCTIONS
# ============================================================================
print("\n" + "=" * 70)
print("7. MATHEMATICAL FUNCTIONS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")

print(f"\nSum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Standard deviation: {np.std(arr)}")
print(f"Variance: {np.var(arr)}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Index of min: {np.argmin(arr)}")
print(f"Index of max: {np.argmax(arr)}")

# 2D operations
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Sum of all elements: {np.sum(arr_2d)}")
print(f"Sum along axis 0 (columns): {np.sum(arr_2d, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(arr_2d, axis=1)}")


# ============================================================================
# 8. BOOLEAN INDEXING AND FILTERING
# ============================================================================
print("\n" + "=" * 70)
print("8. BOOLEAN INDEXING")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

# Boolean conditions
condition = arr > 5
print(f"\nCondition (arr > 5): {condition}")
print(f"Elements > 5: {arr[condition]}")

print(f"\nElements < 5: {arr[arr < 5]}")
print(f"Even numbers: {arr[arr % 2 == 0]}")
print(f"Elements between 3 and 7: {arr[(arr >= 3) & (arr <= 7)]}")

# where function
result = np.where(arr > 5, arr, 0)  # If >5 keep, else 0
print(f"\nwhere(arr > 5, arr, 0): {result}")


# ============================================================================
# 9. ARRAY CONCATENATION AND SPLITTING
# ============================================================================
print("\n" + "=" * 70)
print("9. CONCATENATION AND SPLITTING")
print("=" * 70)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Concatenate
concatenated = np.concatenate([arr1, arr2, arr3])
print(f"Concatenated: {concatenated}")

# Stack vertically (row-wise)
arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])
vstacked = np.vstack([arr_a, arr_b])
print(f"\nVstack:\n{vstacked}")

# Stack horizontally (column-wise)
hstacked = np.hstack([arr_a, arr_b])
print(f"\nHstack: {hstacked}")

# Split
arr = np.arange(12)
split = np.split(arr, 3)  # Split into 3 equal parts
print(f"\nOriginal: {arr}")
print(f"Split into 3: {split}")


# ============================================================================
# 10. COPYING ARRAYS
# ============================================================================
print("\n" + "=" * 70)
print("10. COPYING ARRAYS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Original: {arr}")

# View (shallow copy) - shares data
arr_view = arr.view()
arr_view[0] = 999
print(f"\nAfter modifying view:")
print(f"Original: {arr}")
print(f"View: {arr_view}")

# Copy (deep copy) - independent
arr = np.array([1, 2, 3, 4, 5])
arr_copy = arr.copy()
arr_copy[0] = 999
print(f"\nAfter modifying copy:")
print(f"Original: {arr}")
print(f"Copy: {arr_copy}")


# ============================================================================
# 11. SORTING
# ============================================================================
print("\n" + "=" * 70)
print("11. SORTING")
print("=" * 70)

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Original: {arr}")
print(f"Sorted: {np.sort(arr)}")

# Sort in place
arr.sort()
print(f"After arr.sort(): {arr}")

# 2D sorting
arr_2d = np.array([[3, 1, 4], [9, 2, 6]])
print(f"\n2D array:\n{arr_2d}")
print(f"Sort each row:\n{np.sort(arr_2d, axis=1)}")
print(f"Sort each column:\n{np.sort(arr_2d, axis=0)}")


# ============================================================================
# 12. UNIQUE VALUES
# ============================================================================
print("\n" + "=" * 70)
print("12. UNIQUE VALUES")
print("=" * 70)

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(f"Array: {arr}")
print(f"Unique values: {np.unique(arr)}")

unique, counts = np.unique(arr, return_counts=True)
print(f"Unique with counts: {dict(zip(unique, counts))}")


# ============================================================================
# 13. BROADCASTING
# ============================================================================
print("\n" + "=" * 70)
print("13. BROADCASTING")
print("=" * 70)

# Broadcasting allows operations on arrays of different shapes
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D Array:\n{arr}")

# Add scalar (broadcasts to all elements)
result = arr + 10
print(f"\nAdd 10:\n{result}")

# Add 1D array to 2D array (broadcasts along rows)
arr_1d = np.array([10, 20, 30])
result = arr + arr_1d
print(f"\nAdd [10, 20, 30] (broadcasts to each row):\n{result}")

# Broadcasting with different shapes
arr1 = np.array([[1], [2], [3]])  # 3x1
arr2 = np.array([10, 20, 30, 40])  # 1x4
result = arr1 + arr2  # Results in 3x4
print(f"\nBroadcasting 3x1 + 1x4:\n{result}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("NUMPY BASICS COVERED:")
print("=" * 70)
topics = [
    "1. Creating Arrays (list, arange, linspace, zeros, ones, random)",
    "2. Array Attributes (shape, ndim, size, dtype)",
    "3. Data Types (int, float, bool, type conversion)",
    "4. Indexing and Slicing (1D and 2D)",
    "5. Reshaping (reshape, flatten, transpose)",
    "6. Array Operations (arithmetic, element-wise)",
    "7. Mathematical Functions (sum, mean, min, max, etc.)",
    "8. Boolean Indexing (filtering with conditions)",
    "9. Concatenation and Splitting (stack, split)",
    "10. Copying (view vs copy)",
    "11. Sorting (sort, argsort)",
    "12. Unique Values (unique, counts)",
    "13. Broadcasting (operations on different shapes)"
]

for topic in topics:
    print(f"✓ {topic}")

print("\n" + "=" * 70)
print("All NumPy basics covered successfully!")
print("=" * 70)
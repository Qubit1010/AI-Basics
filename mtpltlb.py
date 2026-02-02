"""
MATPLOTLIB BASICS - COMPLETE BEGINNER'S GUIDE
==============================================
Covering all fundamental plotting and visualization concepts

Topics covered:
- Basic line plots
- Multiple plots and subplots
- Scatter plots
- Bar charts
- Histograms
- Pie charts
- Customization (colors, labels, titles, legends)
- Saving figures
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files

print("Matplotlib version:", matplotlib.__version__)
print("=" * 80)

# ============================================================================
# 1. BASIC LINE PLOT
# ============================================================================
print("\n" + "=" * 80)
print("1. BASIC LINE PLOT")
print("=" * 80)

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Basic Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)

# Save the figure
plt.savefig('plot_01_basic_line.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created basic line plot: plot_01_basic_line.png")


# ============================================================================
# 2. MULTIPLE LINES ON ONE PLOT
# ============================================================================
print("\n" + "=" * 80)
print("2. MULTIPLE LINES ON ONE PLOT")
print("=" * 80)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.plot(x, y3, label='sin(x)*cos(x)', linewidth=2)

plt.title('Multiple Functions', fontsize=16, fontweight='bold')
plt.xlabel('X axis', fontsize=12)
plt.ylabel('Y axis', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.savefig('plot_02_multiple_lines.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created multiple lines plot: plot_02_multiple_lines.png")


# ============================================================================
# 3. LINE STYLES AND COLORS
# ============================================================================
print("\n" + "=" * 80)
print("3. LINE STYLES, COLORS, AND MARKERS")
print("=" * 80)

x = np.linspace(0, 10, 20)

plt.figure(figsize=(12, 8))

# Different line styles
plt.subplot(2, 2, 1)
plt.plot(x, x, '-', label='solid')
plt.plot(x, x+1, '--', label='dashed')
plt.plot(x, x+2, '-.', label='dash-dot')
plt.plot(x, x+3, ':', label='dotted')
plt.title('Line Styles')
plt.legend()
plt.grid(True, alpha=0.3)

# Different colors
plt.subplot(2, 2, 2)
plt.plot(x, x, 'r-', label='red')
plt.plot(x, x+1, 'g-', label='green')
plt.plot(x, x+2, 'b-', label='blue')
plt.plot(x, x+3, 'c-', label='cyan')
plt.plot(x, x+4, 'm-', label='magenta')
plt.title('Colors')
plt.legend()
plt.grid(True, alpha=0.3)

# Different markers
plt.subplot(2, 2, 3)
plt.plot(x, x, 'o-', label='circle')
plt.plot(x, x+1, 's-', label='square')
plt.plot(x, x+2, '^-', label='triangle')
plt.plot(x, x+3, 'D-', label='diamond')
plt.title('Markers')
plt.legend()
plt.grid(True, alpha=0.3)

# Combined styling
plt.subplot(2, 2, 4)
plt.plot(x, x, 'ro-', linewidth=2, markersize=8, label='red circles')
plt.plot(x, x+1, 'bs--', linewidth=2, markersize=8, label='blue squares')
plt.plot(x, x+2, 'g^-.', linewidth=2, markersize=8, label='green triangles')
plt.title('Combined Styling')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_03_styles_colors.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created line styles plot: plot_03_styles_colors.png")


# ============================================================================
# 4. SCATTER PLOT
# ============================================================================
print("\n" + "=" * 80)
print("4. SCATTER PLOTS")
print("=" * 80)

# Generate random data
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = np.random.randint(20, 200, 100)

plt.figure(figsize=(12, 5))

# Basic scatter
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6)
plt.title('Basic Scatter Plot', fontsize=14, fontweight='bold')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)

# Scatter with colors and sizes
plt.subplot(1, 2, 2)
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Color value')
plt.title('Scatter with Colors & Sizes', fontsize=14, fontweight='bold')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_04_scatter.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created scatter plot: plot_04_scatter.png")


# ============================================================================
# 5. BAR CHARTS
# ============================================================================
print("\n" + "=" * 80)
print("5. BAR CHARTS")
print("=" * 80)

categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(12, 5))

# Vertical bar chart
plt.subplot(1, 2, 1)
bars = plt.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
plt.title('Vertical Bar Chart', fontsize=14, fontweight='bold')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}', ha='center', va='bottom', fontsize=10)

# Horizontal bar chart
plt.subplot(1, 2, 2)
plt.barh(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
plt.title('Horizontal Bar Chart', fontsize=14, fontweight='bold')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plot_05_bar_charts.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created bar charts: plot_05_bar_charts.png")


# ============================================================================
# 6. GROUPED AND STACKED BAR CHARTS
# ============================================================================
print("\n" + "=" * 80)
print("6. GROUPED AND STACKED BAR CHARTS")
print("=" * 80)

categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
product_a = [20, 35, 30, 35, 27]
product_b = [25, 32, 34, 20, 25]
product_c = [15, 18, 22, 25, 30]

x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(12, 5))

# Grouped bar chart
plt.subplot(1, 2, 1)
plt.bar(x - width, product_a, width, label='Product A', color='#FF6B6B')
plt.bar(x, product_b, width, label='Product B', color='#4ECDC4')
plt.bar(x + width, product_c, width, label='Product C', color='#45B7D1')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Grouped Bar Chart', fontsize=14, fontweight='bold')
plt.xticks(x, categories)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Stacked bar chart
plt.subplot(1, 2, 2)
plt.bar(categories, product_a, label='Product A', color='#FF6B6B')
plt.bar(categories, product_b, bottom=product_a, label='Product B', color='#4ECDC4')
plt.bar(categories, product_c, bottom=np.array(product_a)+np.array(product_b),
        label='Product C', color='#45B7D1')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Stacked Bar Chart', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot_06_grouped_stacked_bars.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created grouped/stacked bars: plot_06_grouped_stacked_bars.png")


# ============================================================================
# 7. HISTOGRAMS
# ============================================================================
print("\n" + "=" * 80)
print("7. HISTOGRAMS")
print("=" * 80)

# Generate random data
np.random.seed(42)
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(120, 20, 1000)

plt.figure(figsize=(12, 5))

# Basic histogram
plt.subplot(1, 2, 1)
plt.hist(data1, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Basic Histogram', fontsize=14, fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.3)

# Multiple histograms
plt.subplot(1, 2, 2)
plt.hist(data1, bins=30, alpha=0.5, label='Dataset 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Dataset 2', color='red')
plt.title('Multiple Histograms', fontsize=14, fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot_07_histograms.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created histograms: plot_07_histograms.png")


# ============================================================================
# 8. PIE CHARTS
# ============================================================================
print("\n" + "=" * 80)
print("8. PIE CHARTS")
print("=" * 80)

labels = ['Python', 'Java', 'JavaScript', 'C++', 'Others']
sizes = [35, 25, 20, 12, 8]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
explode = (0.1, 0, 0, 0, 0)  # Explode the 1st slice

plt.figure(figsize=(12, 5))

# Basic pie chart
plt.subplot(1, 2, 1)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Basic Pie Chart', fontsize=14, fontweight='bold')

# Exploded pie chart with customization
plt.subplot(1, 2, 2)
wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     explode=explode, shadow=True)
# Make percentage text bold and white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.title('Exploded Pie Chart', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_08_pie_charts.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created pie charts: plot_08_pie_charts.png")


# ============================================================================
# 9. SUBPLOTS - MULTIPLE PLOTS IN ONE FIGURE
# ============================================================================
print("\n" + "=" * 80)
print("9. SUBPLOTS")
print("=" * 80)

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Sine wave
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Wave', fontweight='bold')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cosine wave
axes[0, 1].plot(x, np.cos(x), 'r-', linewidth=2)
axes[0, 1].set_title('Cosine Wave', fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Exponential
axes[1, 0].plot(x, np.exp(x/5), 'g-', linewidth=2)
axes[1, 0].set_title('Exponential', fontweight='bold')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('exp(x/5)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Logarithm
axes[1, 1].plot(x[1:], np.log(x[1:]), 'm-', linewidth=2)
axes[1, 1].set_title('Logarithm', fontweight='bold')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('log(x)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Multiple Subplots Example', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('plot_09_subplots.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created subplots: plot_09_subplots.png")


# ============================================================================
# 10. CUSTOMIZATION - TITLES, LABELS, LEGENDS
# ============================================================================
print("\n" + "=" * 80)
print("10. CUSTOMIZATION")
print("=" * 80)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(12, 6))

plt.plot(x, y1, 'b-', linewidth=2.5, label='Sine wave', alpha=0.8)
plt.plot(x, y2, 'r--', linewidth=2.5, label='Cosine wave', alpha=0.8)

# Title with custom properties
plt.title('Customized Plot Example', fontsize=18, fontweight='bold',
          color='darkblue', pad=20)

# Axis labels
plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold', color='darkgreen')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold', color='darkgreen')

# Legend with custom location and style
plt.legend(loc='upper right', fontsize=12, framealpha=0.9,
          shadow=True, borderpad=1)

# Grid customization
plt.grid(True, linestyle='--', alpha=0.5, color='gray')

# Axis limits
plt.xlim(-0.5, 10.5)
plt.ylim(-1.5, 1.5)

# Add text annotation
plt.text(5, 0.5, 'Peak', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Add arrow annotation
plt.annotate('First crossing', xy=(np.pi/2, 0), xytext=(2, -0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red')

plt.tight_layout()
plt.savefig('plot_10_customization.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created customization example: plot_10_customization.png")


# ============================================================================
# 11. BOX PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("11. BOX PLOTS")
print("=" * 80)

# Generate random data
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

plt.figure(figsize=(10, 6))
box = plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                  patch_artist=True, showmeans=True)

# Customize box colors
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Box Plot Example', fontsize=14, fontweight='bold')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.grid(axis='y', alpha=0.3)

plt.savefig('plot_11_boxplot.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created box plot: plot_11_boxplot.png")


# ============================================================================
# 12. AREA PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("12. AREA PLOTS")
print("=" * 80)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(12, 5))

# Single area plot
plt.subplot(1, 2, 1)
plt.fill_between(x, y1, alpha=0.5, color='blue', label='sin(x)')
plt.plot(x, y1, 'b-', linewidth=2)
plt.title('Single Area Plot', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

# Stacked area plot
plt.subplot(1, 2, 2)
plt.fill_between(x, 0, y1, alpha=0.5, color='blue', label='sin(x)')
plt.fill_between(x, y1, y1+np.abs(y2), alpha=0.5, color='red', label='cos(x)')
plt.title('Stacked Area Plot', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_12_area_plots.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created area plots: plot_12_area_plots.png")


# ============================================================================
# 13. HEATMAPS
# ============================================================================
print("\n" + "=" * 80)
print("13. HEATMAPS")
print("=" * 80)

# Create sample data
np.random.seed(42)
data = np.random.rand(10, 10)

plt.figure(figsize=(10, 8))
im = plt.imshow(data, cmap='YlOrRd', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Intensity', rotation=270, labelpad=20)

# Add labels
plt.title('Heatmap Example', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('X axis')
plt.ylabel('Y axis')

# Add value annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = plt.text(j, i, f'{data[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('plot_13_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created heatmap: plot_13_heatmap.png")


# ============================================================================
# 14. POLAR PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("14. POLAR PLOTS")
print("=" * 80)

theta = np.linspace(0, 2*np.pi, 100)
r = np.abs(np.sin(3*theta))

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, r, linewidth=2, color='blue')
ax.fill(theta, r, alpha=0.3, color='blue')
ax.set_title('Polar Plot Example', fontsize=14, fontweight='bold', pad=20)

plt.savefig('plot_14_polar.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created polar plot: plot_14_polar.png")


# ============================================================================
# 15. 3D PLOTS (BONUS)
# ============================================================================
print("\n" + "=" * 80)
print("15. 3D PLOTS")
print("=" * 80)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# 3D Line plot
ax1 = fig.add_subplot(121, projection='3d')
t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t
ax1.plot(x, y, z, linewidth=2, color='blue')
ax1.set_title('3D Line Plot', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 3D Surface plot
ax2 = fig.add_subplot(122, projection='3d')
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax2.set_title('3D Surface Plot', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig.colorbar(surf, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('plot_15_3d_plots.png', dpi=100, bbox_inches='tight')
plt.close()

print("✓ Created 3D plots: plot_15_3d_plots.png")


# ============================================================================
# 16. FIGURE SIZE AND DPI
# ============================================================================
print("\n" + "=" * 80)
print("16. FIGURE SIZE AND DPI SETTINGS")
print("=" * 80)

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Small figure
plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title('Small Figure (6x4 inches)')
plt.savefig('plot_16_small.png', dpi=100, bbox_inches='tight')
plt.close()

# Large figure with high DPI
plt.figure(figsize=(12, 8))
plt.plot(x, y, linewidth=3)
plt.title('Large Figure (12x8 inches, 150 DPI)', fontsize=16)
plt.savefig('plot_16_large.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Created size examples: plot_16_small.png and plot_16_large.png")


# ============================================================================
# SUMMARY AND FILE LIST
# ============================================================================
print("\n" + "=" * 80)
print("MATPLOTLIB BASICS COVERED:")
print("=" * 80)

topics = [
    "1. Basic Line Plots",
    "2. Multiple Lines",
    "3. Line Styles, Colors, and Markers",
    "4. Scatter Plots",
    "5. Bar Charts (Vertical and Horizontal)",
    "6. Grouped and Stacked Bar Charts",
    "7. Histograms",
    "8. Pie Charts",
    "9. Subplots (Multiple plots in one figure)",
    "10. Customization (Titles, Labels, Legends, Annotations)",
    "11. Box Plots",
    "12. Area Plots",
    "13. Heatmaps",
    "14. Polar Plots",
    "15. 3D Plots",
    "16. Figure Size and DPI Settings"
]

for topic in topics:
    print(f"✓ {topic}")

print("\n" + "=" * 80)
print("FILES GENERATED:")
print("=" * 80)

files = [
    "plot_01_basic_line.png",
    "plot_02_multiple_lines.png",
    "plot_03_styles_colors.png",
    "plot_04_scatter.png",
    "plot_05_bar_charts.png",
    "plot_06_grouped_stacked_bars.png",
    "plot_07_histograms.png",
    "plot_08_pie_charts.png",
    "plot_09_subplots.png",
    "plot_10_customization.png",
    "plot_11_boxplot.png",
    "plot_12_area_plots.png",
    "plot_13_heatmap.png",
    "plot_14_polar.png",
    "plot_15_3d_plots.png",
    "plot_16_small.png",
    "plot_16_large.png"
]

for i, file in enumerate(files, 1):
    print(f"{i:2}. {file}")

print("\n" + "=" * 80)
print("All Matplotlib basics covered successfully!")
print("Total plots created: 17")
print("=" * 80)
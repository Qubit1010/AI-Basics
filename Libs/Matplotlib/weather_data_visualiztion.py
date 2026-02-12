"""
WEATHER DATA VISUALIZATION DASHBOARD
=====================================
A practical Matplotlib project analyzing and visualizing weather data

Features:
- Generate realistic weather data for a full year
- Temperature trends and patterns
- Precipitation analysis
- Humidity and wind patterns
- Seasonal comparisons
- Multiple interactive visualizations
- Comprehensive weather report
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("=" * 80)
print("WEATHER DATA VISUALIZATION DASHBOARD")
print("=" * 80)

# ============================================================================
# 1. GENERATE WEATHER DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. GENERATING WEATHER DATA")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range (full year)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
num_days = len(dates)

print(f"\nGenerating data for {num_days} days")
print(f"Period: {start_date.date()} to {end_date.date()}")

# Generate temperature data with seasonal variation
# Base temperature with seasonal sine wave
day_of_year = np.arange(num_days)
seasonal_temp = 15 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peaks in summer

# Add daily variation
daily_variation = np.random.normal(0, 3, num_days)
max_temp = seasonal_temp + 5 + daily_variation
min_temp = seasonal_temp - 5 + daily_variation * 0.8

# Ensure min < max
for i in range(num_days):
    if min_temp[i] >= max_temp[i]:
        min_temp[i] = max_temp[i] - 2

avg_temp = (max_temp + min_temp) / 2

# Generate precipitation (mm) - more rain in winter/spring
base_precipitation = 2 + 3 * np.abs(np.sin(2 * np.pi * (day_of_year + 60) / 365))
precipitation = np.random.poisson(base_precipitation, num_days).astype(float)
# Add some heavy rain days
heavy_rain_days = np.random.choice(num_days, size=20, replace=False)
precipitation[heavy_rain_days] += np.random.uniform(20, 50, 20)

# Generate humidity (%)
base_humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year + 30) / 365)
humidity = base_humidity + np.random.normal(0, 10, num_days)
humidity = np.clip(humidity, 20, 100)  # Keep within 20-100%

# Generate wind speed (km/h) - windier in winter
base_wind = 15 + 10 * np.abs(np.sin(2 * np.pi * (day_of_year + 180) / 365))
wind_speed = base_wind + np.random.exponential(5, num_days)
wind_speed = np.clip(wind_speed, 0, 60)

# Create DataFrame
weather_df = pd.DataFrame({
    'Date': dates,
    'Max_Temp': max_temp,
    'Min_Temp': min_temp,
    'Avg_Temp': avg_temp,
    'Precipitation': precipitation,
    'Humidity': humidity,
    'Wind_Speed': wind_speed
})

# Add month and season
weather_df['Month'] = weather_df['Date'].dt.month
weather_df['Month_Name'] = weather_df['Date'].dt.strftime('%B')
weather_df['Season'] = pd.cut(weather_df['Month'], bins=[0, 3, 6, 9, 12],
                               labels=['Winter', 'Spring', 'Summer', 'Fall'])

print("\n✓ Weather data generated successfully")
print("\nFirst 5 days:")
print(weather_df[['Date', 'Max_Temp', 'Min_Temp', 'Precipitation', 'Humidity', 'Wind_Speed']].head())

# Save to CSV
weather_df.to_csv('weather_data.csv', index=False)
print("\n✓ Data saved to: weather_data.csv")


# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("2. WEATHER STATISTICS")
print("=" * 80)

print("\n📊 Temperature Statistics:")
print(f"  Average Temperature:    {weather_df['Avg_Temp'].mean():.1f}°C")
print(f"  Highest Temperature:    {weather_df['Max_Temp'].max():.1f}°C on {weather_df.loc[weather_df['Max_Temp'].idxmax(), 'Date'].date()}")
print(f"  Lowest Temperature:     {weather_df['Min_Temp'].min():.1f}°C on {weather_df.loc[weather_df['Min_Temp'].idxmin(), 'Date'].date()}")
print(f"  Temperature Range:      {weather_df['Max_Temp'].max() - weather_df['Min_Temp'].min():.1f}°C")

print("\n🌧️ Precipitation Statistics:")
print(f"  Total Rainfall:         {weather_df['Precipitation'].sum():.1f} mm")
print(f"  Average Daily Rain:     {weather_df['Precipitation'].mean():.1f} mm")
print(f"  Rainy Days (>1mm):      {(weather_df['Precipitation'] > 1).sum()} days")
print(f"  Heaviest Rainfall:      {weather_df['Precipitation'].max():.1f} mm on {weather_df.loc[weather_df['Precipitation'].idxmax(), 'Date'].date()}")

print("\n💨 Wind Statistics:")
print(f"  Average Wind Speed:     {weather_df['Wind_Speed'].mean():.1f} km/h")
print(f"  Maximum Wind Speed:     {weather_df['Wind_Speed'].max():.1f} km/h")
print(f"  Calm Days (<5 km/h):    {(weather_df['Wind_Speed'] < 5).sum()} days")

print("\n💧 Humidity Statistics:")
print(f"  Average Humidity:       {weather_df['Humidity'].mean():.1f}%")
print(f"  Highest Humidity:       {weather_df['Humidity'].max():.1f}%")
print(f"  Lowest Humidity:        {weather_df['Humidity'].min():.1f}%")


# ============================================================================
# 3. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("3. CREATING VISUALIZATIONS")
print("=" * 80)

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')

# -------- VISUALIZATION 1: Temperature Trends --------
print("\n📈 Creating temperature trends visualization...")

fig, ax = plt.subplots(figsize=(16, 8))

# Plot max and min temperatures
ax.plot(weather_df['Date'], weather_df['Max_Temp'], color='#FF6B6B',
        linewidth=1.5, label='Max Temperature', alpha=0.8)
ax.plot(weather_df['Date'], weather_df['Min_Temp'], color='#4ECDC4',
        linewidth=1.5, label='Min Temperature', alpha=0.8)

# Fill between max and min
ax.fill_between(weather_df['Date'], weather_df['Min_Temp'], weather_df['Max_Temp'],
                alpha=0.3, color='gray', label='Temperature Range')

# Plot average
ax.plot(weather_df['Date'], weather_df['Avg_Temp'], color='#FFD93D',
        linewidth=2, label='Average Temperature', linestyle='--')

# Formatting
ax.set_title('Daily Temperature Variations Throughout 2024',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Format x-axis to show months
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Add horizontal line at 0°C
ax.axhline(y=0, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Freezing Point')

plt.tight_layout()
plt.savefig('weather_viz_1_temperature_trends.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: weather_viz_1_temperature_trends.png")


# -------- VISUALIZATION 2: Precipitation Analysis --------
print("\n🌧️ Creating precipitation visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Daily precipitation bar chart
colors = ['#3498db' if p < 10 else '#2ecc71' if p < 30 else '#e74c3c'
          for p in weather_df['Precipitation']]
ax1.bar(weather_df['Date'], weather_df['Precipitation'], color=colors,
        alpha=0.7, width=1, edgecolor='none')
ax1.set_title('Daily Precipitation Throughout 2024', fontsize=16, fontweight='bold')
ax1.set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', alpha=0.7, label='Light Rain (<10mm)'),
    Patch(facecolor='#2ecc71', alpha=0.7, label='Moderate Rain (10-30mm)'),
    Patch(facecolor='#e74c3c', alpha=0.7, label='Heavy Rain (>30mm)')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Monthly total precipitation
monthly_precip = weather_df.groupby('Month_Name')['Precipitation'].sum()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_precip = monthly_precip.reindex([m for m in month_order if m in monthly_precip.index])

bars = ax2.bar(range(len(monthly_precip)), monthly_precip.values,
               color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
ax2.set_title('Monthly Total Precipitation', fontsize=16, fontweight='bold')
ax2.set_ylabel('Total Precipitation (mm)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(monthly_precip)))
ax2.set_xticklabels([m[:3] for m in monthly_precip.index], rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}mm', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('weather_viz_2_precipitation.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: weather_viz_2_precipitation.png")


# -------- VISUALIZATION 3: Multi-variable Dashboard --------
print("\n📊 Creating multi-variable dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Temperature distribution histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(weather_df['Avg_Temp'], bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
ax1.axvline(weather_df['Avg_Temp'].mean(), color='darkred', linestyle='--',
            linewidth=2, label=f"Mean: {weather_df['Avg_Temp'].mean():.1f}°C")
ax1.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Temperature (°C)', fontsize=11)
ax1.set_ylabel('Frequency (days)', fontsize=11)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Humidity over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(weather_df['Date'], weather_df['Humidity'], color='#4ECDC4',
         linewidth=1, alpha=0.8)
ax2.fill_between(weather_df['Date'], weather_df['Humidity'], alpha=0.3, color='#4ECDC4')
ax2.set_title('Humidity Levels Throughout Year', fontsize=14, fontweight='bold')
ax2.set_ylabel('Humidity (%)', fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
ax2.grid(True, alpha=0.3)

# 3. Wind speed over time
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(weather_df['Date'], weather_df['Wind_Speed'], color='#95E1D3',
         linewidth=1, alpha=0.8)
ax3.fill_between(weather_df['Date'], weather_df['Wind_Speed'], alpha=0.3, color='#95E1D3')
ax3.set_title('Wind Speed Throughout Year', fontsize=14, fontweight='bold')
ax3.set_ylabel('Wind Speed (km/h)', fontsize=11)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Seasonal comparison (box plot)
ax4 = fig.add_subplot(gs[1, 1])
seasonal_data = [weather_df[weather_df['Season'] == season]['Avg_Temp'].values
                 for season in ['Winter', 'Spring', 'Summer', 'Fall']]
bp = ax4.boxplot(seasonal_data, labels=['Winter', 'Spring', 'Summer', 'Fall'],
                 patch_artist=True, showmeans=True)
colors_box = ['#AED6F1', '#ABEBC6', '#F9E79F', '#F5B7B1']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax4.set_title('Temperature by Season', fontsize=14, fontweight='bold')
ax4.set_ylabel('Temperature (°C)', fontsize=11)
ax4.grid(axis='y', alpha=0.3)

# 5. Correlation scatter: Temperature vs Humidity
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(weather_df['Avg_Temp'], weather_df['Humidity'],
                     c=weather_df['Precipitation'], cmap='Blues',
                     alpha=0.6, s=30)
ax5.set_title('Temperature vs Humidity (colored by precipitation)',
              fontsize=14, fontweight='bold')
ax5.set_xlabel('Average Temperature (°C)', fontsize=11)
ax5.set_ylabel('Humidity (%)', fontsize=11)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Precipitation (mm)', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Weather conditions pie chart
ax6 = fig.add_subplot(gs[2, 1])
# Categorize days
sunny = ((weather_df['Precipitation'] < 1) & (weather_df['Avg_Temp'] > 15)).sum()
rainy = (weather_df['Precipitation'] > 5).sum()
cloudy = ((weather_df['Precipitation'] <= 5) & (weather_df['Precipitation'] >= 1)).sum()
cold = ((weather_df['Precipitation'] < 1) & (weather_df['Avg_Temp'] <= 15)).sum()

weather_conditions = [sunny, rainy, cloudy, cold]
labels = [f'Sunny\n({sunny} days)', f'Rainy\n({rainy} days)',
          f'Cloudy\n({cloudy} days)', f'Cold\n({cold} days)']
colors_pie = ['#FFD93D', '#6C5CE7', '#A8A8A8', '#74B9FF']
explode = (0.05, 0.05, 0, 0)

ax6.pie(weather_conditions, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, explode=explode)
ax6.set_title('Weather Conditions Distribution', fontsize=14, fontweight='bold')

plt.suptitle('Weather Analysis Dashboard 2024', fontsize=20, fontweight='bold', y=0.995)
plt.savefig('weather_viz_3_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: weather_viz_3_dashboard.png")


# -------- VISUALIZATION 4: Monthly Comparison --------
print("\n📅 Creating monthly comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Monthly average temperature
monthly_temp = weather_df.groupby('Month_Name')['Avg_Temp'].mean()
monthly_temp = monthly_temp.reindex(month_order)
axes[0, 0].plot(range(12), monthly_temp.values, marker='o', markersize=10,
                linewidth=3, color='#FF6B6B')
axes[0, 0].fill_between(range(12), monthly_temp.values, alpha=0.3, color='#FF6B6B')
axes[0, 0].set_title('Average Temperature by Month', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0, 0].set_xticks(range(12))
axes[0, 0].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Monthly precipitation
axes[0, 1].bar(range(12), monthly_precip.values, color='#3498db', alpha=0.8)
axes[0, 1].set_title('Total Precipitation by Month', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Precipitation (mm)', fontsize=11)
axes[0, 1].set_xticks(range(12))
axes[0, 1].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Monthly average wind speed
monthly_wind = weather_df.groupby('Month_Name')['Wind_Speed'].mean()
monthly_wind = monthly_wind.reindex(month_order)
axes[1, 0].plot(range(12), monthly_wind.values, marker='s', markersize=10,
                linewidth=3, color='#95E1D3')
axes[1, 0].fill_between(range(12), monthly_wind.values, alpha=0.3, color='#95E1D3')
axes[1, 0].set_title('Average Wind Speed by Month', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Wind Speed (km/h)', fontsize=11)
axes[1, 0].set_xticks(range(12))
axes[1, 0].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Monthly average humidity
monthly_humidity = weather_df.groupby('Month_Name')['Humidity'].mean()
monthly_humidity = monthly_humidity.reindex(month_order)
axes[1, 1].plot(range(12), monthly_humidity.values, marker='^', markersize=10,
                linewidth=3, color='#4ECDC4')
axes[1, 1].fill_between(range(12), monthly_humidity.values, alpha=0.3, color='#4ECDC4')
axes[1, 1].set_title('Average Humidity by Month', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Humidity (%)', fontsize=11)
axes[1, 1].set_xticks(range(12))
axes[1, 1].set_xticklabels([m[:3] for m in month_order], rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Monthly Weather Patterns 2024', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('weather_viz_4_monthly_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: weather_viz_4_monthly_comparison.png")


# -------- VISUALIZATION 5: Heatmap Calendar --------
print("\n🗓️ Creating temperature heatmap calendar...")

# Create a matrix for the calendar heatmap
# Reshape data by week
weather_df['Week'] = weather_df['Date'].dt.isocalendar().week
weather_df['DayOfWeek'] = weather_df['Date'].dt.dayofweek

# Create pivot table
heatmap_data = weather_df.pivot_table(values='Avg_Temp',
                                       index='Week',
                                       columns='DayOfWeek',
                                       aggfunc='mean')

fig, ax = plt.subplots(figsize=(14, 20))
im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(7))
ax.set_yticks(np.arange(len(heatmap_data)))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_yticklabels([f'Week {i+1}' for i in range(len(heatmap_data))])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Temperature (°C)', rotation=270, labelpad=20, fontsize=12)

# Add title
ax.set_title('Daily Temperature Calendar Heatmap 2024',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('weather_viz_5_heatmap_calendar.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: weather_viz_5_heatmap_calendar.png")


# ============================================================================
# 4. GENERATE WEATHER REPORT
# ============================================================================
print("\n" + "=" * 80)
print("4. GENERATING WEATHER REPORT")
print("=" * 80)

report = f"""
{'='*80}
ANNUAL WEATHER REPORT 2024
{'='*80}

PERIOD: {start_date.date()} to {end_date.date()} ({num_days} days)

{'='*80}
TEMPERATURE ANALYSIS
{'='*80}

Annual Statistics:
  • Average Temperature:           {weather_df['Avg_Temp'].mean():.1f}°C
  • Highest Temperature:           {weather_df['Max_Temp'].max():.1f}°C ({weather_df.loc[weather_df['Max_Temp'].idxmax(), 'Date'].strftime('%B %d')})
  • Lowest Temperature:            {weather_df['Min_Temp'].min():.1f}°C ({weather_df.loc[weather_df['Min_Temp'].idxmin(), 'Date'].strftime('%B %d')})
  • Temperature Range:             {weather_df['Max_Temp'].max() - weather_df['Min_Temp'].min():.1f}°C

Seasonal Averages:
  • Winter (Jan-Mar):              {weather_df[weather_df['Season']=='Winter']['Avg_Temp'].mean():.1f}°C
  • Spring (Apr-Jun):              {weather_df[weather_df['Season']=='Spring']['Avg_Temp'].mean():.1f}°C
  • Summer (Jul-Sep):              {weather_df[weather_df['Season']=='Summer']['Avg_Temp'].mean():.1f}°C
  • Fall (Oct-Dec):                {weather_df[weather_df['Season']=='Fall']['Avg_Temp'].mean():.1f}°C

Temperature Days:
  • Hot Days (>25°C):              {(weather_df['Max_Temp'] > 25).sum()} days
  • Mild Days (15-25°C):           {((weather_df['Avg_Temp'] >= 15) & (weather_df['Avg_Temp'] <= 25)).sum()} days
  • Cold Days (<15°C):             {(weather_df['Avg_Temp'] < 15).sum()} days
  • Freezing Days (<0°C):          {(weather_df['Min_Temp'] < 0).sum()} days

{'='*80}
PRECIPITATION ANALYSIS
{'='*80}

Annual Statistics:
  • Total Precipitation:           {weather_df['Precipitation'].sum():.1f} mm
  • Average Daily Precipitation:   {weather_df['Precipitation'].mean():.1f} mm
  • Wettest Day:                   {weather_df['Precipitation'].max():.1f} mm ({weather_df.loc[weather_df['Precipitation'].idxmax(), 'Date'].strftime('%B %d')})
  • Wettest Month:                 {monthly_precip.idxmax()} ({monthly_precip.max():.1f} mm)

Precipitation Days:
  • Total Rainy Days (>1mm):       {(weather_df['Precipitation'] > 1).sum()} days
  • Light Rain (1-5mm):            {((weather_df['Precipitation'] > 1) & (weather_df['Precipitation'] <= 5)).sum()} days
  • Moderate Rain (5-15mm):        {((weather_df['Precipitation'] > 5) & (weather_df['Precipitation'] <= 15)).sum()} days
  • Heavy Rain (>15mm):            {(weather_df['Precipitation'] > 15).sum()} days
  • Dry Days (<1mm):               {(weather_df['Precipitation'] < 1).sum()} days

{'='*80}
WIND ANALYSIS
{'='*80}

Wind Statistics:
  • Average Wind Speed:            {weather_df['Wind_Speed'].mean():.1f} km/h
  • Maximum Wind Speed:            {weather_df['Wind_Speed'].max():.1f} km/h ({weather_df.loc[weather_df['Wind_Speed'].idxmax(), 'Date'].strftime('%B %d')})
  • Calm Days (<5 km/h):           {(weather_df['Wind_Speed'] < 5).sum()} days
  • Breezy Days (5-20 km/h):       {((weather_df['Wind_Speed'] >= 5) & (weather_df['Wind_Speed'] < 20)).sum()} days
  • Windy Days (>20 km/h):         {(weather_df['Wind_Speed'] >= 20).sum()} days

{'='*80}
HUMIDITY ANALYSIS
{'='*80}

Humidity Statistics:
  • Average Humidity:              {weather_df['Humidity'].mean():.1f}%
  • Highest Humidity:              {weather_df['Humidity'].max():.1f}% ({weather_df.loc[weather_df['Humidity'].idxmax(), 'Date'].strftime('%B %d')})
  • Lowest Humidity:               {weather_df['Humidity'].min():.1f}% ({weather_df.loc[weather_df['Humidity'].idxmin(), 'Date'].strftime('%B %d')})

Humidity Levels:
  • Very Humid (>80%):             {(weather_df['Humidity'] > 80).sum()} days
  • Humid (60-80%):                {((weather_df['Humidity'] >= 60) & (weather_df['Humidity'] <= 80)).sum()} days
  • Comfortable (40-60%):          {((weather_df['Humidity'] >= 40) & (weather_df['Humidity'] < 60)).sum()} days
  • Dry (<40%):                    {(weather_df['Humidity'] < 40).sum()} days

{'='*80}
WEATHER CONDITIONS SUMMARY
{'='*80}

  • Sunny Days:                    {sunny} days ({sunny/num_days*100:.1f}%)
  • Rainy Days:                    {rainy} days ({rainy/num_days*100:.1f}%)
  • Cloudy Days:                   {cloudy} days ({cloudy/num_days*100:.1f}%)
  • Cold Days:                     {cold} days ({cold/num_days*100:.1f}%)

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)

# Save report
with open('weather_report.txt', 'w') as f:
    f.write(report)

print("✓ Report saved to: weather_report.txt")


# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("WEATHER ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Visualizations Created:")
visualizations = [
    "1. weather_viz_1_temperature_trends.png - Daily temperature variations",
    "2. weather_viz_2_precipitation.png - Daily and monthly precipitation",
    "3. weather_viz_3_dashboard.png - Multi-variable analysis dashboard",
    "4. weather_viz_4_monthly_comparison.png - Monthly weather patterns",
    "5. weather_viz_5_heatmap_calendar.png - Temperature calendar heatmap"
]
for viz in visualizations:
    print(f"  ✓ {viz}")

print("\n📁 Files Generated:")
files = [
    "weather_data.csv - Raw weather data (366 days)",
    "weather_report.txt - Comprehensive weather report",
    "5 PNG visualization files"
]
for file in files:
    print(f"  • {file}")

print("\n📈 Key Findings:")
findings = [
    f"Average temperature: {weather_df['Avg_Temp'].mean():.1f}°C",
    f"Total rainfall: {weather_df['Precipitation'].sum():.1f} mm",
    f"Rainy days: {(weather_df['Precipitation'] > 1).sum()} days",
    f"Wettest month: {monthly_precip.idxmax()}",
    f"Hottest day: {weather_df['Max_Temp'].max():.1f}°C",
    f"Weather condition: {sunny} sunny, {rainy} rainy, {cloudy} cloudy, {cold} cold days"
]
for finding in findings:
    print(f"  • {finding}")

print("\n" + "=" * 80)
print("All weather data analyzed and visualized successfully!")
print("=" * 80)
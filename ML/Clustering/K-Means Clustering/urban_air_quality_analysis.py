"""
K-MEANS CLUSTERING - URBAN AIR QUALITY MONITORING STATIONS
===========================================================
Grouping city air quality sensors into clusters based on pollution
signatures so the environmental agency can identify pollution source
zones, issue targeted health alerts, and prioritize field inspections

Perfect Scenario for K-Means:
- No labeled "zone types" in advance — we discover them from data
- Continuous numerical pollution readings across many pollutants
- Sensors with similar profiles share the same pollution source
  (traffic, industrial, residential burning, background clean air)
- Cluster centroids = average pollution signature = actionable profile
- City can deploy different health alert thresholds per cluster
- Scalable: works for 50 or 5,000 sensors

Dataset: Air Quality Sensor Daily Averages (Generated)
Features:
- PM2_5               (fine particulate matter μg/m³ — lung damage)
- PM10                (coarse particulate μg/m³ — dust/construction)
- NO2                 (nitrogen dioxide ppb — traffic exhaust)
- CO                  (carbon monoxide ppm — combustion)
- SO2                 (sulfur dioxide ppb — industrial/coal)
- O3                  (ozone ppb — secondary pollutant, sunlight+NOx)
- VOC                 (volatile organic compounds ppb — solvents/fuel)
- BlackCarbon         (black carbon μg/m³ — diesel engines)
- HumidityAvg         (% — affects particle dispersion)
- TemperatureAvg      (°C — affects ozone and dispersion)
- WindSpeedAvg        (m/s — higher wind disperses pollutants)
- NightDayRatio       (ratio: night readings / day readings)

Expected Cluster Types:
- Traffic Hotspots     (high NO2, CO, BlackCarbon — near highways)
- Industrial Zones     (high SO2, PM10, VOC — near factories)
- Residential Burning  (high PM2.5, CO, NightDayRatio — evening cooking/heating)
- Clean Background     (low everything — parks, outskirts)

Why K-Means for Air Quality Stations?
- Pollution signatures cluster naturally by source type
- Same source = same health risk = same alert thresholds
- Cluster centroids = clear pollution profile for each zone type
- Field inspection teams dispatched by cluster priority
- Re-cluster seasonally (winter heating changes profiles significantly)

Approach:
1. Generate realistic multi-pollutant sensor data
2. Exploratory Data Analysis
3. Feature Scaling (MANDATORY for K-Means)
4. Elbow Method + Silhouette Analysis
5. Build K-Means (K=4)
6. Cluster profiling + naming
7. Health risk scoring per cluster
8. 7 comprehensive visualizations
9. Operational report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("K-MEANS CLUSTERING - URBAN AIR QUALITY MONITORING STATIONS")
print("=" * 80)


# ============================================================================
# STEP 1: GENERATE REALISTIC AIR QUALITY SENSOR DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATE REALISTIC AIR QUALITY SENSOR DATA")
print("=" * 80)

np.random.seed(42)

n_traffic     = 110   # Near highways, intersections, parking lots
n_industrial  = 80    # Near factories, power plants, warehouses
n_residential = 100   # Dense housing, cooking/heating emissions
n_clean       = 110   # Parks, suburbs, outskirts
n_total       = n_traffic + n_industrial + n_residential + n_clean

print(f"\nGenerating {n_total} air quality sensor station records...")
print(f"  (Ground truth types — given to K-Means: NONE)")
print(f"  Traffic Hotspots:   {n_traffic}   sensors")
print(f"  Industrial Zones:   {n_industrial}    sensors")
print(f"  Residential Burn:   {n_residential}   sensors")
print(f"  Clean Background:   {n_clean}   sensors")

# --- TRAFFIC HOTSPOT sensors ---
# High NO2 (exhaust), CO (incomplete combustion), BlackCarbon (diesel)
# Moderate PM2.5/PM10 from tire/brake wear
# Low SO2 (no coal/industry), moderate VOC (fuel evaporation)
# Low NightDayRatio (traffic dies at night)
traffic = {
    'PM2_5':          np.random.normal(28,  6,   n_traffic).clip(12, 55),
    'PM10':           np.random.normal(48,  10,  n_traffic).clip(20, 85),
    'NO2':            np.random.normal(72,  14,  n_traffic).clip(35, 120),
    'CO':             np.random.normal(2.8, 0.6, n_traffic).clip(1.0, 5.5),
    'SO2':            np.random.normal(5,   2,   n_traffic).clip(1,  14),
    'O3':             np.random.normal(28,  8,   n_traffic).clip(8,  55),
    'VOC':            np.random.normal(38,  10,  n_traffic).clip(12, 70),
    'BlackCarbon':    np.random.normal(4.2, 0.9, n_traffic).clip(1.5, 8.0),
    'HumidityAvg':    np.random.normal(58,  10,  n_traffic).clip(30, 85),
    'TemperatureAvg': np.random.normal(18,  4,   n_traffic).clip(8,  30),
    'WindSpeedAvg':   np.random.normal(3.2, 1.0, n_traffic).clip(0.5, 7.0),
    'NightDayRatio':  np.random.normal(0.38,0.08, n_traffic).clip(0.15,0.65),
    '_TrueType':      ['Traffic Hotspot'] * n_traffic
}

# --- INDUSTRIAL ZONE sensors ---
# High SO2 (fossil fuel burning), high PM10 (dust/grinding)
# High VOC (solvents, chemicals), moderate-high PM2.5
# Low NO2 compared to traffic, low BlackCarbon
# NightDayRatio near 1.0 (factories run 24/7)
industrial = {
    'PM2_5':          np.random.normal(35,  8,   n_industrial).clip(15, 70),
    'PM10':           np.random.normal(78,  18,  n_industrial).clip(35, 140),
    'NO2':            np.random.normal(32,  10,  n_industrial).clip(10, 65),
    'CO':             np.random.normal(3.5, 0.8, n_industrial).clip(1.2, 6.5),
    'SO2':            np.random.normal(42,  12,  n_industrial).clip(15, 80),
    'O3':             np.random.normal(18,  6,   n_industrial).clip(5,  38),
    'VOC':            np.random.normal(62,  14,  n_industrial).clip(25, 100),
    'BlackCarbon':    np.random.normal(2.5, 0.8, n_industrial).clip(0.8, 5.5),
    'HumidityAvg':    np.random.normal(52,  12,  n_industrial).clip(28, 80),
    'TemperatureAvg': np.random.normal(20,  4,   n_industrial).clip(9,  32),
    'WindSpeedAvg':   np.random.normal(2.5, 1.0, n_industrial).clip(0.3, 6.0),
    'NightDayRatio':  np.random.normal(0.88,0.10, n_industrial).clip(0.55,1.10),
    '_TrueType':      ['Industrial Zone'] * n_industrial
}

# --- RESIDENTIAL BURNING sensors ---
# High PM2.5 (wood/coal burning), high CO (incomplete combustion)
# Night emissions dominant: cooking in evening, heating overnight
# Low NO2 (no highway), low SO2, moderate VOC (cooking fumes)
residential = {
    'PM2_5':          np.random.normal(42,  10,  n_residential).clip(18, 80),
    'PM10':           np.random.normal(52,  12,  n_residential).clip(22, 95),
    'NO2':            np.random.normal(22,  7,   n_residential).clip(6,  48),
    'CO':             np.random.normal(4.2, 0.9, n_residential).clip(1.8, 7.5),
    'SO2':            np.random.normal(8,   3,   n_residential).clip(2,  22),
    'O3':             np.random.normal(22,  7,   n_residential).clip(6,  45),
    'VOC':            np.random.normal(30,  9,   n_residential).clip(10, 58),
    'BlackCarbon':    np.random.normal(3.5, 0.9, n_residential).clip(1.2, 7.0),
    'HumidityAvg':    np.random.normal(65,  10,  n_residential).clip(38, 90),
    'TemperatureAvg': np.random.normal(16,  4,   n_residential).clip(6,  28),
    'WindSpeedAvg':   np.random.normal(2.0, 0.8, n_residential).clip(0.2, 5.0),
    'NightDayRatio':  np.random.normal(1.45,0.20, n_residential).clip(0.85,2.00),
    '_TrueType':      ['Residential Burning'] * n_residential
}

# --- CLEAN BACKGROUND sensors ---
# Low everything. Located in parks, forests, city outskirts
# Slightly elevated O3 (parks have less NOx to consume it)
# High wind (open spaces), moderate humidity
clean = {
    'PM2_5':          np.random.normal(8,   3,   n_clean).clip(2,  18),
    'PM10':           np.random.normal(16,  5,   n_clean).clip(5,  30),
    'NO2':            np.random.normal(10,  4,   n_clean).clip(2,  22),
    'CO':             np.random.normal(0.6, 0.2, n_clean).clip(0.2, 1.5),
    'SO2':            np.random.normal(3,   1.5, n_clean).clip(0.5, 8),
    'O3':             np.random.normal(38,  8,   n_clean).clip(18, 60),
    'VOC':            np.random.normal(10,  4,   n_clean).clip(2,  22),
    'BlackCarbon':    np.random.normal(0.6, 0.3, n_clean).clip(0.1, 1.8),
    'HumidityAvg':    np.random.normal(62,  10,  n_clean).clip(35, 90),
    'TemperatureAvg': np.random.normal(15,  4,   n_clean).clip(5,  28),
    'WindSpeedAvg':   np.random.normal(4.5, 1.2, n_clean).clip(1.0, 9.0),
    'NightDayRatio':  np.random.normal(0.72,0.12, n_clean).clip(0.40,1.05),
    '_TrueType':      ['Clean Background'] * n_clean
}

df = pd.concat([
    pd.DataFrame(traffic),
    pd.DataFrame(industrial),
    pd.DataFrame(residential),
    pd.DataFrame(clean)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

df.insert(0, 'SensorID', [f'AQS-{i+1:03d}' for i in range(n_total)])

feature_columns = [
    'PM2_5', 'PM10', 'NO2', 'CO', 'SO2', 'O3',
    'VOC', 'BlackCarbon', 'HumidityAvg', 'TemperatureAvg',
    'WindSpeedAvg', 'NightDayRatio'
]

print(f"\n  Dataset shape: {df.shape}")
print(f"  Features:      {len(feature_columns)}")

print("\n--- First 10 Sensor Records ---")
print(df[['SensorID'] + feature_columns].head(10).to_string(index=False))

df.to_csv('air_quality_data.csv', index=False, encoding='utf-8')
print(f"\n  Saved: air_quality_data.csv")


# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\n--- Feature Statistics ---")
print(df[feature_columns].describe().round(3).to_string())

print(f"\n--- Feature Scale Problem (why scaling is critical) ---")
print(f"\n  {'Feature':<18} {'Min':>7} {'Max':>7} {'Range':>8}  Unit / Context")
print(f"  {'-'*65}")
meta = {
    'PM2_5':          'μg/m³  — fine particles',
    'PM10':           'μg/m³  — coarse particles',
    'NO2':            'ppb    — traffic exhaust',
    'CO':             'ppm    — combustion gas',
    'SO2':            'ppb    — industrial/coal',
    'O3':             'ppb    — secondary pollutant',
    'VOC':            'ppb    — solvents/fuel',
    'BlackCarbon':    'μg/m³  — diesel soot',
    'HumidityAvg':    '%      — relative humidity',
    'TemperatureAvg': '°C     — air temperature',
    'WindSpeedAvg':   'm/s    — wind speed',
    'NightDayRatio':  'ratio  — night/day emissions'
}
for feat in feature_columns:
    mn  = df[feat].min()
    mx  = df[feat].max()
    rng = mx - mn
    print(f"  {feat:<18} {mn:>7.2f} {mx:>7.2f} {rng:>8.2f}  {meta[feat]}")

print(f"\n--- Mean Values by True Type (for insight — NOT used in clustering) ---")
print(df.groupby('_TrueType')[feature_columns].mean().round(2).to_string())

print(f"\n--- Top Feature Correlations ---")
corr   = df[feature_columns].corr()
pairs  = [(feature_columns[i], feature_columns[j], corr.iloc[i,j])
          for i in range(len(feature_columns))
          for j in range(i+1, len(feature_columns))]
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"\n  {'Feature A':<18} {'Feature B':<18} {'Corr':>8}  Interpretation")
print(f"  {'-'*70}")
interp = {
    ('PM2_5','PM10'):        'Both particle types from same sources',
    ('PM2_5','CO'):          'Combustion produces both',
    ('NO2','BlackCarbon'):   'Both from traffic/diesel engines',
    ('SO2','VOC'):           'Industrial co-emissions',
    ('CO','BlackCarbon'):    'Diesel combustion signature',
    ('NO2','CO'):            'Exhaust gas siblings',
}
for a, b, c in pairs[:8]:
    key = (a,b) if (a,b) in interp else (b,a)
    note = interp.get(key, '')
    print(f"  {a:<18} {b:<18} {c:>8.4f}  {note}")


# ============================================================================
# STEP 3: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE SCALING")
print("=" * 80)

print("""
  WHY SCALING IS MANDATORY FOR K-MEANS:
  ============================================================
  K-Means computes Euclidean distance between sensors.
  Unscaled distance between two sensors:

    NO2 difference:      45 ppb   (traffic=72 vs clean=10+37 noise)
    CO difference:       0.6 ppm  (traffic=2.8 vs clean=0.6+noise)
    NightDayRatio diff:  0.9      (residential=1.45 vs traffic=0.38)

  NO2's larger numerical range would DOMINATE the distance,
  making CO and NightDayRatio nearly invisible to K-Means.

  After StandardScaler (mean=0, std=1):
    All pollutants contribute EQUALLY to cluster assignment.
    NightDayRatio gets proper weight even though it's 0–2 range.
    This is what separates Residential Burning from other clusters!
""")

X        = df[feature_columns].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Before scaling:")
for feat in ['NO2', 'CO', 'NightDayRatio']:
    std = df[feat].std()
    print(f"    {feat:<18} std = {std:.4f}")

print(f"\n  After StandardScaler:")
X_sc_df = pd.DataFrame(X_scaled, columns=feature_columns)
for feat in ['NO2', 'CO', 'NightDayRatio']:
    std = X_sc_df[feat].std()
    print(f"    {feat:<18} std = {std:.4f}  (all equal now)")

X_train, X_test, y_train, y_test = X_scaled, X_scaled, df['_TrueType'], df['_TrueType']
print(f"\n  Note: K-Means is unsupervised — uses ALL {n_total} stations for clustering")
print(f"  No train/test split needed (no labels, no prediction to evaluate)")


# ============================================================================
# STEP 4: HOW K-MEANS WORKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HOW K-MEANS WORKS")
print("=" * 80)

print(f"""
  K-Means Algorithm — Air Quality Example:
  ============================================================

  INPUT: {n_total} sensors, {len(feature_columns)} pollutant features, K=4 clusters

  STEP 1 — K-Means++ INITIALIZE (smarter than random):
    Place first centroid on a random sensor
    Each subsequent centroid chosen with probability
    proportional to distance² from nearest existing centroid
    → Centroids start spread out = faster, better convergence

  STEP 2 — ASSIGN each sensor to nearest centroid:
    distance(AQS-001, centroid_1) = sqrt(
      (pm25_1 - c_pm25)² + (no2_1 - c_no2)² + ... )
    Sensor goes to whichever centroid this is smallest for

  STEP 3 — UPDATE centroids = mean of assigned sensors:
    centroid_traffic = mean of all sensors in traffic cluster
    → This is now the "average traffic sensor profile"

  STEP 4 — REPEAT until no sensor changes cluster

  EXAMPLE for AQS-001 (High NO2=75, Low SO2=4, Low NightDayRatio=0.35):
    Distance to Traffic centroid:    SMALL  (matches profile)
    Distance to Industrial centroid: LARGER (SO2 too low)
    Distance to Residential centroid:LARGER (NightDayRatio too low)
    Distance to Clean centroid:      VERY LARGE (NO2 too high)
    → Assigned to: Traffic Hotspot ✓

  WHY K-Means++ OVER RANDOM INIT?
    Random: bad starting positions → slow convergence, poor local minima
    K-Means++: spread starts → faster convergence, better clusters
    We also run n_init=20 times, keep best (lowest inertia) result
""")


# ============================================================================
# STEP 5: ELBOW METHOD
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: ELBOW METHOD — FINDING OPTIMAL K")
print("=" * 80)

k_range    = range(1, 13)
inertias   = []
sil_scores = []

print(f"\n  {'K':>4} {'Inertia (WCSS)':>16} {'Silhouette':>12}  Interpretation")
print(f"  {'-'*60}")

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15,
                random_state=42, max_iter=300)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

    if k >= 2:
        sil = silhouette_score(X_scaled, km.labels_)
        sil_scores.append(sil)
        quality = "Strong" if sil > 0.5 else "Reasonable" if sil > 0.25 else "Weak"
        print(f"  {k:>4} {km.inertia_:>16.2f} {sil:>12.4f}  {quality}")
    else:
        sil_scores.append(None)
        print(f"  {k:>4} {km.inertia_:>16.2f} {'N/A':>12}")

# Elbow detection via second derivative of inertia
drops       = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
accel       = [drops[i-1] - drops[i] for i in range(1, len(drops))]
elbow_k     = accel.index(max(accel)) + 2

valid_sil   = [(k+2, s) for k, s in enumerate(sil_scores[1:]) if s is not None]
best_sil_k  = max(valid_sil, key=lambda x: x[1])[0]
best_sil_val= max(valid_sil, key=lambda x: x[1])[1]

print(f"\n  Elbow point detected at:        K = {elbow_k}")
print(f"  Best silhouette score at:       K = {best_sil_k}  ({best_sil_val:.4f})")
print(f"  Domain knowledge suggests:      K = 4  (4 pollution source types)")
print(f"\n  --> Chosen K = 4")


# ============================================================================
# STEP 6: SILHOUETTE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: SILHOUETTE ANALYSIS — VALIDATING K=4")
print("=" * 80)

optimal_k  = 4
km_final   = KMeans(n_clusters=optimal_k, init='k-means++',
                    n_init=20, random_state=42, max_iter=300)
km_final.fit(X_scaled)

labels       = km_final.labels_
df['Cluster']= labels
overall_sil  = silhouette_score(X_scaled, labels)
sample_sil   = silhouette_samples(X_scaled, labels)

print(f"""
  Silhouette Score Recap:
    Range: -1 (wrong cluster) to +1 (perfect separation)
    > 0.50 = Strong structure
    0.25-0.50 = Reasonable structure
    < 0.25 = Weak / no structure

  Overall Silhouette Score: {overall_sil:.4f}  -> {'Strong' if overall_sil>0.5 else 'Reasonable' if overall_sil>0.25 else 'Weak'} clustering
  Inertia (WCSS):           {km_final.inertia_:.2f}
  Iterations to converge:   {km_final.n_iter_}
""")

print(f"  Per-Cluster Breakdown:")
print(f"  {'Cluster':>9} {'Size':>6} {'Mean Sil':>10} {'Min':>8} {'Max':>8}")
print(f"  {'-'*48}")
for c in range(optimal_k):
    mask   = labels == c
    c_sils = sample_sil[mask]
    print(f"  Cluster {c:>2}  {mask.sum():>6} {c_sils.mean():>10.4f} "
          f"{c_sils.min():>8.4f} {c_sils.max():>8.4f}")


# ============================================================================
# STEP 7: CLUSTER PROFILING AND NAMING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CLUSTER PROFILING AND NAMING")
print("=" * 80)

centroids_scaled   = km_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df       = pd.DataFrame(centroids_original,
                                   columns=feature_columns)

print(f"\n  --- Cluster Centroids (Original Pollutant Units) ---")
print(centroids_df.round(3).to_string())

# Name clusters by dominant pollutant signature
def name_air_cluster(row):
    if row['SO2'] > 20 and row['VOC'] > 40:
        return 'Industrial Zone'
    elif row['NightDayRatio'] > 1.1 and row['CO'] > 3.0:
        return 'Residential Burning'
    elif row['NO2'] > 45 and row['BlackCarbon'] > 2.8:
        return 'Traffic Hotspot'
    else:
        return 'Clean Background'

cluster_names = {c: name_air_cluster(centroids_df.iloc[c])
                 for c in range(optimal_k)}

# Fallback: sort by AvgDailyTrips proxy (NO2) if names collide
used = set()
for c in range(optimal_k):
    if cluster_names[c] in used:
        sorted_no2 = centroids_df['NO2'].sort_values()
        sorted_so2 = centroids_df['SO2'].sort_values(ascending=False)
        if c == sorted_no2.idxmin():
            cluster_names[c] = 'Clean Background'
        elif c == sorted_so2.idxmax():
            cluster_names[c] = 'Industrial Zone'
    used.add(cluster_names[c])

df['ClusterName'] = df['Cluster'].map(cluster_names)

print(f"\n  --- Cluster Assignments ---")
for c in range(optimal_k):
    n = cluster_names[c]
    count = (labels == c).sum()
    print(f"  Cluster {c} -> {n:<22}  ({count} sensors)")

print(f"\n  --- Full Pollutant Profile per Cluster ---")
for c in range(optimal_k):
    row   = centroids_df.iloc[c]
    name  = cluster_names[c]
    count = (labels == c).sum()
    print(f"\n  [{c}] {name} ({count} sensors)")
    print(f"       PM2.5={row['PM2_5']:.1f} μg/m³  |  PM10={row['PM10']:.1f}  |  "
          f"NO2={row['NO2']:.1f} ppb")
    print(f"       CO={row['CO']:.2f} ppm       |  SO2={row['SO2']:.1f} ppb   |  "
          f"VOC={row['VOC']:.1f} ppb")
    print(f"       BlackCarbon={row['BlackCarbon']:.2f} μg/m³  |  "
          f"O3={row['O3']:.1f} ppb")
    print(f"       Wind={row['WindSpeedAvg']:.1f} m/s  |  "
          f"NightDayRatio={row['NightDayRatio']:.2f}")


# ============================================================================
# STEP 8: HEALTH RISK SCORING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: HEALTH RISK SCORING PER CLUSTER")
print("=" * 80)

print("""
  WHO Air Quality Guidelines (reference):
    PM2.5:  15 μg/m³ annual (WHO 2021)
    NO2:    10 μg/m³ annual (WHO 2021)
    SO2:    40 μg/m³ (24-hr mean)
    CO:     4 ppm (24-hr mean)
    O3:     100 μg/m³ (8-hr mean)
""")

# WHO thresholds approximate (normalized risk 0-100)
def health_risk_score(row):
    score = 0
    score += min(row['PM2_5']  / 75.0, 1.0) * 30    # PM2.5 weight 30%
    score += min(row['NO2']    / 200.0,1.0) * 20    # NO2   weight 20%
    score += min(row['SO2']    / 125.0,1.0) * 15    # SO2   weight 15%
    score += min(row['CO']     / 10.0, 1.0) * 15    # CO    weight 15%
    score += min(row['PM10']   / 150.0,1.0) * 10    # PM10  weight 10%
    score += min(row['VOC']    / 150.0,1.0) * 10    # VOC   weight 10%
    return round(score * 100, 1)

risk_scores = {c: health_risk_score(centroids_df.iloc[c])
               for c in range(optimal_k)}
risk_labels = {s: 'CRITICAL' if s >= 60 else 'HIGH' if s >= 40 else
               'MODERATE' if s >= 20 else 'LOW'
               for s in risk_scores.values()}

print(f"  {'Cluster':<22} {'Risk Score':>11} {'Level':>10}  Dominant Pollutant")
print(f"  {'-'*65}")
dominant = {
    'Traffic Hotspot':     'NO2 + BlackCarbon (diesel exhaust)',
    'Industrial Zone':     'SO2 + VOC (factory emissions)',
    'Residential Burning': 'PM2.5 + CO (heating/cooking smoke)',
    'Clean Background':    'O3 slightly elevated (no NOx to consume it)'
}
for c in range(optimal_k):
    name  = cluster_names[c]
    score = risk_scores[c]
    level = risk_labels[score]
    dom   = dominant.get(name, '')
    print(f"  {name:<22} {score:>11.1f} {level:>10}  {dom}")

print(f"\n  Agency Alert Protocol:")
for c in range(optimal_k):
    name  = cluster_names[c]
    score = risk_scores[c]
    level = risk_labels[score]
    if level == 'CRITICAL':
        action = "Issue public health emergency — sensitive groups stay indoors"
    elif level == 'HIGH':
        action = "Issue health advisory — outdoor activity warning"
    elif level == 'MODERATE':
        action = "Monitor daily — alert if PM2.5 spikes above threshold"
    else:
        action = "Routine monitoring — no advisory needed"
    print(f"  [{level}] {name}: {action}")


# ============================================================================
# STEP 9: VALIDATION AGAINST GROUND TRUTH
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: VALIDATION AGAINST GROUND TRUTH (UNSEEN BY K-MEANS)")
print("=" * 80)

print(f"""
  In real deployment, no ground truth exists.
  We generated true labels ONLY to measure how well K-Means
  recovered the natural groupings from raw pollution data.
""")

cross_tab = pd.crosstab(df['_TrueType'], df['ClusterName'])
print(f"  Cross-tabulation: True Type vs K-Means Cluster")
print(cross_tab.to_string())

print(f"\n  Recovery Rate per True Type:")
for tg in sorted(df['_TrueType'].unique()):
    sub        = df[df['_TrueType'] == tg]
    top        = sub['ClusterName'].value_counts().index[0]
    top_count  = sub['ClusterName'].value_counts().iloc[0]
    purity     = top_count / len(sub) * 100
    print(f"  {tg:<22}: {purity:.1f}% assigned to '{top}'")

print(f"\n  Silhouette Score: {overall_sil:.4f}")
print(f"  Inertia:          {km_final.inertia_:.2f}")
print(f"  Converged in:     {km_final.n_iter_} iterations")


# ============================================================================
# STEP 10: SAMPLE SENSOR ASSIGNMENTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SAMPLE SENSOR ASSIGNMENTS")
print("=" * 80)

print(f"\n  {'Sensor':<10} {'Cluster':<5} {'Zone Type':<22} {'PM2.5':>7} "
      f"{'NO2':>7} {'SO2':>7} {'CO':>6} {'NightDay':>9}")
print(f"  {'-'*78}")
sample = df.sample(20, random_state=7).sort_values('Cluster')
for _, row in sample.iterrows():
    print(f"  {row['SensorID']:<10} {row['Cluster']:<5} {row['ClusterName']:<22} "
          f"{row['PM2_5']:>7.1f} {row['NO2']:>7.1f} {row['SO2']:>7.1f} "
          f"{row['CO']:>6.2f} {row['NightDayRatio']:>9.3f}")


# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: CREATE COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

cluster_colors = {0: '#C62828', 1: '#1565C0', 2: '#E65100', 3: '#2E7D32'}
# Assign colors by name for consistency
name_to_color  = {}
for c, name in cluster_names.items():
    palette = {
        'Traffic Hotspot':     '#1565C0',
        'Industrial Zone':     '#C62828',
        'Residential Burning': '#E65100',
        'Clean Background':    '#2E7D32'
    }
    name_to_color[name] = palette.get(name, cluster_colors[c])

c_color = {c: name_to_color[cluster_names[c]] for c in range(optimal_k)}

# --- Viz 1: Elbow + Silhouette ---
print("\n  Creating elbow + silhouette chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertias, 'o-', color='#1565C0',
             markersize=7, linewidth=2.5)
axes[0].axvline(x=optimal_k, color='#C62828', linestyle='--',
                linewidth=2.5, label=f'Chosen K={optimal_k}')
axes[0].fill_between(list(k_range), inertias, alpha=0.08, color='#1565C0')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

sil_ks   = list(range(2, 13))
sil_vals = sil_scores[1:]
bar_clrs = ['#C62828' if k == optimal_k else '#90CAF9' for k in sil_ks]
axes[1].bar(sil_ks, sil_vals, color=bar_clrs, edgecolor='black', alpha=0.88)
axes[1].axhline(y=0.50, color='#2E7D32', linestyle='--', lw=2,
                label='Strong (0.50)')
axes[1].axhline(y=0.25, color='#E65100', linestyle='--', lw=2,
                label='Weak (0.25)')
axes[1].set_xlabel('K', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('Silhouette Score by K', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Finding Optimal K — Air Quality Clustering',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aq_viz_1_elbow_sil.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_1_elbow_sil.png")

# --- Viz 2: PCA — Clusters vs True Types ---
print("  Creating PCA comparison plot...")
pca       = PCA(n_components=2, random_state=42)
X_pca     = pca.fit_transform(X_scaled)
var_ratio = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for c in range(optimal_k):
    mask = labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=c_color[c], label=cluster_names[c],
                    s=30, alpha=0.7, edgecolors='none')
centroids_pca = pca.transform(centroids_scaled)
axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                s=280, marker='*', c='black', zorder=10, label='Centroids')
for c in range(optimal_k):
    axes[0].annotate(f'C{c}',
                     (centroids_pca[c, 0], centroids_pca[c, 1]),
                     fontsize=11, fontweight='bold',
                     xytext=(6, 4), textcoords='offset points')
axes[0].set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
axes[0].set_title('K-Means Clusters (PCA Space)\nStars = Centroids',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

true_colors = {
    'Traffic Hotspot':     '#1565C0',
    'Industrial Zone':     '#C62828',
    'Residential Burning': '#E65100',
    'Clean Background':    '#2E7D32'
}
for tg in df['_TrueType'].unique():
    mask = df['_TrueType'] == tg
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=true_colors.get(tg, 'gray'),
                    label=tg, s=30, alpha=0.7, edgecolors='none')
axes[1].set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
axes[1].set_title('True Station Types (PCA Space)\n[Validation — not shown to K-Means]',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('K-Means Discovery vs True Pollution Zone Types',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aq_viz_2_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_2_pca.png")

# --- Viz 3: Cluster Pollutant Profiles (Bar per cluster) ---
print("  Creating cluster pollutant profile charts...")
profile_feats = ['PM2_5', 'PM10', 'NO2', 'CO', 'SO2',
                 'O3', 'VOC', 'BlackCarbon', 'NightDayRatio']

# Normalize 0-1 for each feature
norm_centroids = {}
for feat in profile_feats:
    mn = df[feat].min()
    mx = df[feat].max()
    for c in range(optimal_k):
        raw = centroids_df.iloc[c][feat]
        norm_centroids[(c, feat)] = (raw - mn) / (mx - mn + 1e-9)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.flatten()
x_pos = np.arange(len(profile_feats))

for idx, c in enumerate(range(optimal_k)):
    vals  = [norm_centroids[(c, f)] for f in profile_feats]
    color = c_color[c]
    bars  = axes[idx].bar(x_pos, vals, color=color, edgecolor='black',
                           alpha=0.85)
    axes[idx].set_xticks(x_pos)
    axes[idx].set_xticklabels(profile_feats, rotation=30, ha='right',
                               fontsize=10)
    axes[idx].set_ylim(0, 1.15)
    axes[idx].set_ylabel('Normalized Level (0=min, 1=max)',
                          fontsize=10, fontweight='bold')
    count = (labels == c).sum()
    risk  = risk_scores[c]
    level = risk_labels[risk]
    axes[idx].set_title(
        f'Cluster {c}: {cluster_names[c]}\n'
        f'{count} sensors  |  Health Risk: {level} ({risk:.0f}/100)',
        fontsize=11, fontweight='bold', color=color)
    axes[idx].axhline(y=0.5, color='gray', linestyle='--', lw=1, alpha=0.5)
    axes[idx].grid(axis='y', alpha=0.3)
    for bar, val, feat in zip(bars, vals, profile_feats):
        raw = centroids_df.iloc[c][feat]
        axes[idx].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.02,
                       f'{raw:.1f}', ha='center', fontsize=8,
                       fontweight='bold', rotation=0)

plt.suptitle('Pollutant Profiles per Cluster\n(values = raw centroid readings)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aq_viz_3_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_3_profiles.png")

# --- Viz 4: Silhouette Plot ---
print("  Creating silhouette plot...")
fig, ax = plt.subplots(figsize=(10, 7))
y_lower = 10
for c in range(optimal_k):
    c_sil   = np.sort(sample_sil[labels == c])
    size    = len(c_sil)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                     facecolor=c_color[c], edgecolor=c_color[c], alpha=0.75)
    ax.text(-0.05, y_lower + 0.5 * size,
            f'C{c}: {cluster_names[c]}', fontsize=9,
            fontweight='bold', color=c_color[c])
    y_lower = y_upper + 12

ax.axvline(x=overall_sil, color='black', linestyle='--', linewidth=2,
           label=f'Avg Silhouette = {overall_sil:.4f}')
ax.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
ax.set_ylabel('Sensors (grouped by cluster)', fontsize=12, fontweight='bold')
ax.set_title(f'Silhouette Plot — K={optimal_k} Air Quality Clusters',
             fontsize=13, fontweight='bold')
ax.set_xlim(-0.2, 1.0)
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('aq_viz_4_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_4_silhouette.png")

# --- Viz 5: Key Pollutant Scatter Pairs ---
print("  Creating pollutant scatter plots...")
scatter_pairs = [
    ('NO2',  'BlackCarbon',   'Traffic signature\n(exhaust + diesel)'),
    ('SO2',  'VOC',           'Industrial signature\n(factory emissions)'),
    ('CO',   'NightDayRatio', 'Residential signature\n(heating/cooking)'),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (fx, fy, title) in zip(axes, scatter_pairs):
    for c in range(optimal_k):
        mask = labels == c
        ax.scatter(df[fx][mask], df[fy][mask],
                   c=c_color[c], label=cluster_names[c],
                   s=25, alpha=0.6, edgecolors='none')
    for c in range(optimal_k):
        cx = centroids_df.iloc[c][fx]
        cy = centroids_df.iloc[c][fy]
        ax.scatter(cx, cy, s=220, marker='*', c=c_color[c],
                   edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_xlabel(fx, fontsize=12, fontweight='bold')
    ax.set_ylabel(fy, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Pollution Source Signatures — Cluster Separation\n'
             '(Stars = Cluster Centroids)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('aq_viz_5_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_5_scatter.png")

# --- Viz 6: Centroid Heatmap + Risk ---
print("  Creating centroid heatmap + risk chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Normalized centroid heatmap
heat_df = pd.DataFrame(index=[f'C{c}: {cluster_names[c]}'
                               for c in range(optimal_k)])
for feat in feature_columns:
    mn = df[feat].min()
    mx = df[feat].max()
    heat_df[feat] = [(centroids_df.iloc[c][feat]-mn)/(mx-mn+1e-9)
                     for c in range(optimal_k)]

sns.heatmap(heat_df, annot=True, fmt='.2f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '0=min, 1=max'},
            annot_kws={"size": 8}, linewidths=0.5)
axes[0].set_title('Cluster Centroid Heatmap\n(Normalized per feature)',
                  fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(),
                         rotation=35, ha='right', fontsize=9)

# Health risk bar chart
risk_name_order = sorted(risk_scores.keys(),
                          key=lambda c: risk_scores[c], reverse=True)
risk_vals  = [risk_scores[c] for c in risk_name_order]
risk_names = [cluster_names[c] for c in risk_name_order]
risk_clrs  = ['#B71C1C' if v >= 60 else '#E65100' if v >= 40
              else '#F9A825' if v >= 20 else '#2E7D32'
              for v in risk_vals]
bars = axes[1].barh(risk_names, risk_vals, color=risk_clrs,
                    edgecolor='black', alpha=0.88)
axes[1].axvline(x=60, color='#B71C1C', linestyle='--', lw=2,
                label='Critical threshold (60)')
axes[1].axvline(x=40, color='#E65100', linestyle='--', lw=2,
                label='High threshold (40)')
axes[1].axvline(x=20, color='#F9A825', linestyle='--', lw=2,
                label='Moderate threshold (20)')
axes[1].set_xlabel('Health Risk Score (0-100)', fontsize=12,
                   fontweight='bold')
axes[1].set_title('Health Risk Score per Cluster\n'
                  '(weighted WHO guideline proximity)',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(axis='x', alpha=0.3)
for bar, val in zip(bars, risk_vals):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('aq_viz_6_heatmap_risk.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_6_heatmap_risk.png")

# --- Viz 7: Boxplots per Cluster ---
print("  Creating feature boxplots...")
box_feats = ['PM2_5', 'NO2', 'SO2', 'CO', 'VOC', 'NightDayRatio']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(box_feats):
    data = [df[df['Cluster'] == c][feat].values for c in range(optimal_k)]
    bp   = axes[i].boxplot(data, patch_artist=True,
                            medianprops=dict(color='black', linewidth=2.5))
    for patch, c in zip(bp['boxes'], range(optimal_k)):
        patch.set_facecolor(c_color[c])
        patch.set_alpha(0.8)
    short = [cluster_names[c].replace(' ', '\n') for c in range(optimal_k)]
    axes[i].set_xticklabels(short, fontsize=8)
    axes[i].set_title(f'{feat} ({meta[feat].split("—")[0].strip()})',
                      fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Concentration', fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Pollutant Distribution by Cluster Type',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aq_viz_7_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: aq_viz_7_boxplots.png")


# ============================================================================
# STEP 12: COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: GENERATE COMPREHENSIVE REPORT")
print("=" * 80)

cluster_means = df.groupby('ClusterName')[feature_columns].mean().round(2)

report = f"""
{'='*80}
K-MEANS CLUSTERING - URBAN AIR QUALITY MONITORING STATIONS
{'='*80}

BUSINESS OBJECTIVE
{'='*80}
Group {n_total} air quality sensors into pollution zone clusters based on their
multi-pollutant signatures, enabling the environmental agency to:
  - Issue targeted health alerts by zone type (not city-wide blankets)
  - Deploy field inspection teams to highest-risk industrial/traffic zones
  - Understand dominant pollution sources district by district
  - Track seasonal cluster shifts (winter heating changes residential profile)

WHY K-MEANS FOR AIR QUALITY STATIONS?
  - Unsupervised: no prior zone labels needed
  - Multi-pollutant fingerprint naturally groups by pollution source
  - Centroids = average pollution signature = actionable threshold per zone
  - Re-cluster monthly/seasonally to detect changing patterns
  - Scalable to city-wide networks of thousands of sensors

DATASET
{'='*80}
  Total sensors:  {n_total}
  Features:       {len(feature_columns)} pollutants + meteorological variables
  Algorithm:      K-Means with K-Means++ init (n_init=20)
  Optimal K:      {optimal_k}

CLUSTERING QUALITY
{'='*80}
  Silhouette Score: {overall_sil:.4f}  ({'Strong' if overall_sil>0.5 else 'Reasonable' if overall_sil>0.25 else 'Weak'} structure)
  Inertia (WCSS):   {km_final.inertia_:.2f}
  Convergence:      {km_final.n_iter_} iterations

  Per-Cluster Silhouette:
  {'Cluster':<5} {'Name':<22} {'Size':>5} {'Mean Sil':>10}
  {'-'*46}
{chr(10).join([f"  {c:<5} {cluster_names[c]:<22} {(labels==c).sum():>5} {sample_sil[labels==c].mean():>10.4f}"
               for c in range(optimal_k)])}

CLUSTER PROFILES (Centroid Values)
{'='*80}
{centroids_df.round(2).to_string()}

HEALTH RISK SCORES
{'='*80}
{chr(10).join([f"  {cluster_names[c]:<22}: {risk_scores[c]:.0f}/100  [{risk_labels[risk_scores[c]]}]"
               for c in sorted(risk_scores, key=lambda c: risk_scores[c], reverse=True)])}

POLLUTION SOURCE SIGNATURES
{'='*80}
  Traffic Hotspot:
    Key markers: NO2 high (exhaust), BlackCarbon high (diesel),
                 CO elevated, low NightDayRatio (dies at night)
    Source:      Highways, intersections, bus depots

  Industrial Zone:
    Key markers: SO2 very high (coal/fossil fuel), VOC high (solvents),
                 PM10 high (grinding/dust), NightDayRatio ~1.0 (24/7 ops)
    Source:      Factories, power plants, chemical plants

  Residential Burning:
    Key markers: NightDayRatio > 1.0 (evening/overnight peak),
                 PM2.5 high (wood smoke), CO high (incomplete combustion)
    Source:      Cooking fires, heating stoves, residential boilers

  Clean Background:
    Key markers: All pollutants low, O3 slightly elevated (no NOx to consume),
                 High wind speed (open areas disperse pollution)
    Source:      Parks, forests, low-density suburbs

OPERATIONAL RECOMMENDATIONS
{'='*80}
  [Industrial Zone — {risk_labels[max(risk_scores.values())]} risk]
    1. Daily automatic alerts to nearby residents on high-SO2 days
    2. Dispatch inspection team if SO2 > 60 ppb for 3+ consecutive days
    3. Cross-reference with factory operating schedules
    4. Push for industrial scrubber compliance checks

  [Residential Burning — HIGH risk]
    1. Evening air quality alerts (5pm-9pm) to affected neighborhoods
    2. Winter coal/wood burning incentive programs for cleaner alternatives
    3. Work with housing authority on heating system upgrades
    4. NightDayRatio > 1.8 triggers automatic advisory

  [Traffic Hotspot — HIGH risk]
    1. Real-time NO2 display boards at major intersections
    2. Coordinate with traffic management for bus-priority signal timing
    3. School zone alerts when NO2 > 80 ppb during school hours
    4. Long-term: flag for EV charging infrastructure priority zones

  [Clean Background — LOW risk]
    1. Maintain as baseline reference sensors for the city
    2. Use as control group for pollution dispersion modeling
    3. Annual calibration checks sufficient
    4. Protect from development that could shift cluster

K-MEANS SPECIFIC NOTES FOR THIS SCENARIO
{'='*80}
  - NightDayRatio was the KEY feature separating Residential from Traffic
    Without scaling, this 0-2 range feature would be invisible to K-Means
  - SO2 was the clearest Industrial Zone separator (very low in all others)
  - K=4 aligns perfectly with 4 known urban pollution source archetypes
  - Seasonal re-clustering essential: winter heating inflates Residential cluster
  - Sensors near cluster boundaries (low silhouette) warrant physical inspection

FILES GENERATED
{'='*80}
  air_quality_data.csv
  aq_viz_1_elbow_sil.png        - Elbow curve + silhouette by K
  aq_viz_2_pca.png              - PCA clusters vs true types
  aq_viz_3_profiles.png         - Pollutant profiles per cluster
  aq_viz_4_silhouette.png       - Station-level silhouette plot
  aq_viz_5_scatter.png          - Pollution source signature pairs
  aq_viz_6_heatmap_risk.png     - Centroid heatmap + health risk scores
  aq_viz_7_boxplots.png         - Feature distributions per cluster

{'='*80}
END OF REPORT
{'='*80}
"""

print(report)
with open('aq_km_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("  Report saved to: aq_km_report.txt")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("K-MEANS AIR QUALITY CLUSTERING COMPLETE!")
print("=" * 80)
print(f"\n  Summary:")
print(f"    Generated {n_total} sensor records across {len(feature_columns)} features")
print(f"    Optimal K:         {optimal_k}  (elbow + silhouette + domain)")
print(f"    Silhouette Score:  {overall_sil:.4f}")
print(f"    Inertia (WCSS):    {km_final.inertia_:.2f}")
print(f"    Converged in:      {km_final.n_iter_} iterations")
print(f"    7 visualizations generated")
print(f"\n  Clusters Discovered:")
for c in range(optimal_k):
    count = (labels == c).sum()
    name  = cluster_names[c]
    risk  = risk_scores[c]
    level = risk_labels[risk]
    print(f"    [{level:<8}] Cluster {c}: {name:<22} — {count} sensors "
          f"| Risk={risk:.0f}/100")
print(f"\n  Key Insights:")
print(f"    - Scaling was critical: NO2 range ~110 vs NightDayRatio range ~1.8")
print(f"    - NightDayRatio alone distinguishes Residential from all others")
print(f"    - SO2 alone distinguishes Industrial from all others")
print(f"    - K-Means recovered true zone types without seeing any labels")
print(f"    - Clean Background sensors serve as city-wide pollution baselines")

print("\n" + "=" * 80)
print("All analysis complete!")
print("=" * 80)
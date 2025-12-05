"""
Final NO2 and O3 Model Training
- O3: Train first (unchanged, already good)
- NO2: Focus all improvements on winter model + blending + peak weighting + residual calibration
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*80)
print("FINAL NO2 AND O3 MODEL TRAINING")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================
print("\n2. Creating features...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)
df['is_weekend_or_holiday'] = df['is_weekend'].copy()

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'monsoon'
    else:
        return 'post_monsoon'

df['season'] = df['month'].apply(get_season)
df['is_winter'] = (df['season'] == 'winter').astype(int)
df['is_summer'] = (df['season'] == 'summer').astype(int)
df['is_monsoon'] = (df['season'] == 'monsoon').astype(int)
df['is_post_monsoon'] = (df['season'] == 'post_monsoon').astype(int)

# Derived meteorological features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
    df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)

# ==================== NO2 FEATURES ====================
print("   Creating NO2-specific features...")

# Extended lags
no2_lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in no2_lag_features:
    if col in df.columns:
        for lag in [1, 3, 6, 12, 24]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# STEP 5: BLH-based lag features
if 'blh_era5' in df.columns:
    df['blh_lag_1h'] = df['blh_era5'].shift(1)
    # blh_rolling_mean_3h will be created later from blh_roll3 to avoid duplication
    if 'no2' in df.columns and 'no2_lag_1h' in df.columns:
        df['blh_no2_lag1_interaction'] = df['blh_era5'] * df['no2_lag_1h']

# PM interactions
if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
    df['pm25_pm10_product'] = df['pm2p5'] * df['pm10']
if 'no2' in df.columns and 'pm2p5' in df.columns:
    df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)
    df['no2_pm25_product'] = df['no2'] * df['pm2p5']

# Wind components
if 'u10_era5' in df.columns:
    df['wind_u'] = df['u10_era5']
    df['wind_u_abs'] = np.abs(df['u10_era5'])
if 'v10_era5' in df.columns:
    df['wind_v'] = df['v10_era5']
    df['wind_v_abs'] = np.abs(df['v10_era5'])
if 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_uv_product'] = df['u10_era5'] * df['v10_era5']

# Rolling means
for window in [3, 6, 12, 24]:
    for feat in ['no2', 'pm2p5', 'pm10']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

# Interactions
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)
if 'blh_era5' in df.columns and 't2m_era5' in df.columns:
    df['blh_temp_interaction'] = df['blh_era5'] * df['t2m_era5']
if 'hour' in df.columns and 't2m_era5' in df.columns:
    df['hour_temp_interaction'] = df['hour'] * df['t2m_era5']

# STEP 1: WINTER-SPECIFIC FEATURES
print("   Creating winter-specific features...")
# Temperature inversion strength
if 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['inversion_strength'] = df['t2m_era5'] - df['d2m_era5']
    df['inversion_strength_abs'] = np.abs(df['inversion_strength'])

# Night flag (STEP 1: exact specification)
df['is_night'] = ((df['hour'] < 7) | (df['hour'] > 20)).astype(int)

# Morning peak (STEP 1: exact specification - hour in [7,8,9])
df['morning_peak'] = df['hour'].isin([7, 8, 9]).astype(int)

# ==================== SEASON-ROBUST FEATURES (High impact for R²) ====================
print("   Creating season-robust features...")

# (A) Inversion strength
if 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['inv'] = df['t2m_era5'] - df['d2m_era5']
    df['inversion_strength'] = df['inv']  # Keep old name for compatibility
    df['inversion_strength_abs'] = np.abs(df['inv'])

# (B) Ventilation index
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['ventilation'] = df['blh_era5'] * df['wind_speed']
    df['blh_wind'] = df['ventilation']  # Keep old name for compatibility
    df['blh_wind_interaction'] = df['ventilation']  # Keep old name for compatibility
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)  # Also keep this

# (C) Stability index (NEW - critical for season-robustness)
if 'inv' in df.columns and 'blh_era5' in df.columns:
    df['stability'] = df['inv'] * (1.0 / (df['blh_era5'] + 1e-6))  # Avoid division by zero
    # Higher stability = stronger inversion + lower BLH = more trapped pollutants

# (D) Traffic proxies
df['hour'] = df['datetime'].dt.hour
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']  # Traffic pattern interaction

# (E) Smooth BLH
if 'blh_era5' in df.columns:
    df['blh_roll3'] = df['blh_era5'].rolling(window=3, min_periods=1).mean()
    df['blh_rolling_mean_3h'] = df['blh_roll3']  # Keep old name for compatibility

# STEP 1: Winter-only interactions (CRITICAL)
# blh_inversion = blh * inversion_strength
if 'inversion_strength' in df.columns and 'blh_era5' in df.columns:
    df['blh_inversion'] = df['blh_era5'] * df['inversion_strength']
    df['blh_inv'] = df['blh_inversion']  # Keep old name for compatibility
# no2lag_blhlag = NO2_target_lag1 * blh
if 'NO2_target' in df.columns:
    df['NO2_target_lag1'] = df['NO2_target'].shift(1)
    if 'blh_era5' in df.columns:
        df['no2lag_blhlag'] = df['NO2_target_lag1'] * df['blh_era5']
        df['no2_blhlag'] = df['no2lag_blhlag']  # Keep old name for compatibility

# STEP 1: Add rolling means for winter
if 't2m_era5' in df.columns:
    df['temperature_roll3'] = df['t2m_era5'].rolling(window=3, min_periods=1).mean()

# STEP 4: Post-monsoon features
print("   Creating post-monsoon-specific features...")
# Stubble burning flag (Oct-Nov)
df['stubble_burning_flag'] = df['month'].isin([10, 11]).astype(int)

# STEP 2: Diwali flag (5 days around Diwali)
# Diwali typically falls in late Oct to early Nov
# Approximate: 5 days around Oct 20-Nov 10 (using a 5-day window)
# For simplicity, using Oct 20-24 and Nov 1-5 as Diwali periods
df['diwali_flag'] = (
    ((df['month'] == 10) & (df['day'] >= 20) & (df['day'] <= 24)) | 
    ((df['month'] == 11) & (df['day'] >= 1) & (df['day'] <= 5))
).astype(int)
df['festival_flag'] = df['diwali_flag'].copy()  # Keep old name for compatibility

# STEP 2: Low wind flag for post-monsoon
if 'wind_speed' in df.columns:
    df['low_wind_flag'] = (df['wind_speed'] < 1.0).astype(int)
    df['wind_stagnation_index'] = df['low_wind_flag']  # Keep old name for compatibility

# STEP 2: Low BLH flag for post-monsoon (blh < 100)
if 'blh_era5' in df.columns:
    df['low_blh_flag'] = (df['blh_era5'] < 100).astype(int)
    df['low_blh'] = df['low_blh_flag']  # Keep old name for compatibility

# O3 features (keep same)
print("   Creating O3-specific photochemical features...")
if 'solar_elevation' in df.columns:
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    df['solar_elevation_squared'] = df['solar_elevation']**2
    df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
    df['solar_elevation_positive'] = np.maximum(0, df['solar_elevation'])
if 'SZA_deg' in df.columns:
    df['sza_rad'] = np.radians(df['SZA_deg'])
    df['cos_sza'] = np.cos(df['sza_rad'])
    df['photolysis_rate_approx'] = np.maximum(0, df['cos_sza'])
if 't2m_era5' in df.columns and 'solar_elevation' in df.columns:
    df['temp_solar_interaction'] = df['t2m_era5'] * np.abs(df['solar_elevation'])
    df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
    df['temp_solar_elevation_squared'] = df['t2m_era5'] * df['solar_elevation_squared']
if 't2m_era5' in df.columns and 'photolysis_rate_approx' in df.columns:
    df['temp_photolysis'] = df['t2m_era5'] * df['photolysis_rate_approx']
if 't2m_era5' in df.columns and 'cos_sza' in df.columns:
    df['temp_cos_sza'] = df['t2m_era5'] * df['cos_sza']
if 'blh_era5' in df.columns:
    if 'solar_elevation' in df.columns:
        df['pbl_solar_elevation'] = df['blh_era5'] * df['solar_elevation_abs']
        df['pbl_solar_elevation_squared'] = df['blh_era5'] * df['solar_elevation_squared']
    if 'photolysis_rate_approx' in df.columns:
        df['pbl_photolysis'] = df['blh_era5'] * df['photolysis_rate_approx']
    if 'cos_sza' in df.columns:
        df['pbl_cos_sza'] = df['blh_era5'] * df['cos_sza']
    if 't2m_era5' in df.columns:
        df['pbl_temp'] = df['blh_era5'] * df['t2m_era5']
if 'wind_speed' in df.columns and 'blh_era5' in df.columns:
    df['pbl_wind_product'] = df['blh_era5'] * df['wind_speed']
if 'relative_humidity_approx' in df.columns and 't2m_era5' in df.columns:
    df['rh_temp_interaction'] = df['relative_humidity_approx'] * df['t2m_era5']
if 'is_weekend' in df.columns and 'solar_elevation_abs' in df.columns:
    df['weekend_solar'] = df['is_weekend'] * df['solar_elevation_abs']
for col in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
    if col in df.columns:
        for lag in [1, 3, 6]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
for window in [3, 6, 12]:
    for feat in ['O3_target', 'no2', 't2m_era5']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
def get_no2_features(df, include_winter_features=True):
    """NO2 feature set with winter-specific features"""
    features = []
    
    # Core pollutants
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no'] if f in df.columns])
    
    # Meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    
    # BLH-based features (STEP 5)
    blh_features = ['blh_lag_1h', 'blh_rolling_mean_3h', 'blh_no2_lag1_interaction']
    features.extend([f for f in blh_features if f in df.columns])
    
    # Extended lags
    for lag in [1, 3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    
    # Rolling means
    for window in [3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    
    # PM interactions
    pm_interactions = ['pm25_pm10_ratio', 'pm25_pm10_product',
                      'no2_pm25_ratio', 'no2_pm25_product']
    features.extend([f for f in pm_interactions if f in df.columns])
    
    # Wind components
    wind_features = ['wind_u', 'wind_v', 'wind_u_abs', 'wind_v_abs', 'wind_uv_product']
    features.extend([f for f in wind_features if f in df.columns])
    
    # Season-robust features (High impact for R²)
    season_robust_features = [
        'inv', 'inversion_strength',  # (A) Inversion strength
        'ventilation', 'blh_wind', 'ventilation_rate',  # (B) Ventilation index
        'stability',  # (C) Stability index (NEW - critical)
        'hour', 'is_weekend', 'hour_weekend_interaction',  # (D) Traffic proxies
        'blh_roll3', 'blh_rolling_mean_3h'  # (E) Smooth BLH
    ]
    features.extend([f for f in season_robust_features if f in df.columns])
    
    # Other interactions
    interaction_features = ['blh_temp_interaction', 'hour_temp_interaction']
    features.extend([f for f in interaction_features if f in df.columns])
    
    # Time features
    time_features = ['month', 'day_of_week', 'is_weekday',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    # Winter-specific features (STEP 1)
    if include_winter_features:
        winter_features = ['inversion_strength', 'inversion_strength_abs', 
                          'is_night', 'morning_peak',
                          'blh_wind', 'blh_wind_interaction',  # blh_wind = blh * wind_speed
                          'blh_inversion', 'blh_inv',  # blh_inversion = blh * inversion_strength
                          'no2lag_blhlag', 'no2_blhlag',  # no2lag_blhlag = NO2_target_lag1 * blh
                          'blh_roll3', 'blh_rolling_mean_3h',  # blh_roll3
                          'temperature_roll3']  # temperature_roll3
        features.extend([f for f in winter_features if f in df.columns])
    
    # Post-monsoon features (STEP 2)
    post_monsoon_features = ['stubble_burning_flag', 
                            'diwali_flag', 'festival_flag',  # diwali_flag (5 days around Diwali)
                            'low_wind_flag', 'wind_stagnation_index',  # low_wind_flag = wind_speed < 1.0
                            'low_blh_flag', 'low_blh']  # low_blh_flag = blh < 100
    features.extend([f for f in post_monsoon_features if f in df.columns])
    
    # Solar
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in features:
        if f in df.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features

def get_o3_features(df):
    """O3 feature set (unchanged)"""
    features = []
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    solar_features = ['solar_elevation', 'solar_elevation_abs', 'solar_elevation_squared',
                     'solar_elevation_positive', 'is_daytime', 'SZA_deg', 'sza_rad',
                     'cos_sza', 'photolysis_rate_approx']
    features.extend([f for f in solar_features if f in df.columns])
    photo_interactions = ['temp_solar_elevation', 'temp_solar_elevation_squared',
                         'temp_photolysis', 'temp_cos_sza']
    features.extend([f for f in photo_interactions if f in df.columns])
    pbl_solar_features = ['pbl_solar_elevation', 'pbl_solar_elevation_squared',
                         'pbl_photolysis', 'pbl_cos_sza', 'pbl_temp']
    features.extend([f for f in pbl_solar_features if f in df.columns])
    other_interactions = ['ventilation_rate', 'pbl_wind_product', 'rh_temp_interaction',
                         'weekend_solar']
    features.extend([f for f in other_interactions if f in df.columns])
    for lag in [1, 3, 6]:
        for feat in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    for window in [3, 6, 12]:
        for feat in ['O3_target', 'no2', 't2m_era5']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in features:
        if f in df.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features

# ==================== DATA PREPARATION ====================
def prepare_data(df, target_col, features, train_mask, val_mask, test_mask):
    """Prepare data"""
    valid_mask = ~df[target_col].isna()
    train_idx = valid_mask & train_mask
    val_idx = valid_mask & val_mask
    test_idx = valid_mask & test_mask
    
    X_train = df[train_idx][features].copy()
    y_train = df[train_idx][target_col].copy()
    X_val = df[val_idx][features].copy()
    y_val = df[val_idx][target_col].copy()
    X_test = df[test_idx][features].copy()
    y_test = df[test_idx][target_col].copy()
    
    # Convert to numeric
    for col in features:
        if col in X_train.columns:
            # Get the column as a Series
            col_series = X_train[col]
            if isinstance(col_series, pd.Series):
                col_dtype = col_series.dtype
            else:
                # If it's somehow a DataFrame, skip it
                continue
            
            if col_dtype == 'object' or str(col_dtype) == 'object':
                X_train[col] = pd.Categorical(X_train[col]).codes
                if col in X_val.columns:
                    X_val[col] = pd.Categorical(X_val[col], categories=pd.Categorical(X_train[col]).categories).codes
                if col in X_test.columns:
                    X_test[col] = pd.Categorical(X_test[col], categories=pd.Categorical(X_train[col]).categories).codes
            elif col_dtype == 'bool' or str(col_dtype) == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype(int)
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype(int)
    
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    features = [f for f in features if f in X_train.columns]
    
    # Fill NaN
    for col in X_train.columns:
        # Ensure we get a scalar value
        col_series = X_train[col]
        if isinstance(col_series, pd.Series):
            null_count = int(col_series.isnull().sum())
        else:
            null_count = 0
        
        if null_count > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            if col in X_val.columns:
                X_val[col].fillna(median_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(median_val, inplace=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, features

# ==================== STEP 3: PEAK-WEIGHTED TRAINING ====================
def calculate_sample_weights(y, percentile=75, weight_factor=4.0):
    """Calculate sample weights: higher weight for high-pollution events"""
    threshold = np.percentile(y, percentile)
    weights = np.ones(len(y))
    weights[y > threshold] = weight_factor
    return weights

# ==================== STEP 4: RESIDUAL CALIBRATION ====================
def train_residual_calibrator(y_true, y_pred, method='isotonic'):
    """Train residual calibrator"""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred, y_true)
    else:  # linear
        calibrator = LinearRegression()
        calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator

# ==================== TRAIN O3 MODEL (FIRST, UNCHANGED) ====================
def train_o3_model(df, train_mask, val_mask, test_mask):
    """Train O3 model (unchanged, already good)"""
    print(f"\n{'='*80}")
    print("TRAINING O3 MODEL (UNCHANGED)")
    print(f"{'='*80}")
    
    features = get_o3_features(df)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, 'O3_target', features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred)
    }
    val_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred),
        'R2': r2_score(y_val, y_val_pred)
    }
    test_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== TRAIN NO2 MODEL (GLOBAL ONLY) ====================
def train_no2_model(df, train_mask, val_mask, test_mask):
    """Train NO2 global model only"""
    print(f"\n{'='*80}")
    print("TRAINING NO2 GLOBAL MODEL")
    print(f"{'='*80}")
    
    features = get_no2_features(df, include_winter_features=True)
    
    print("\n   Training global NO2 model...")
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, 'NO2_target', features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Peak-weighted training
    sample_weights = calculate_sample_weights(y_train, percentile=75, weight_factor=4.0)
    print(f"   Using peak-weighted loss (4.0x weight for >75th percentile)")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Train residual calibrator on validation set
    print("   Training residual calibrator on validation set...")
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    calibrator = train_residual_calibrator(y_val, val_pred, method='isotonic')
    
    # Calculate train and val metrics (before calibration)
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
        'MAE': mean_absolute_error(y_train, train_pred),
        'R2': r2_score(y_train, train_pred)
    }
    val_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_val, val_pred)),
        'MAE': mean_absolute_error(y_val, val_pred),
        'R2': r2_score(y_val, val_pred)
    }
    
    # Apply calibration to test set
    test_pred_calibrated = calibrator.predict(test_pred)
    
    test_metrics_uncalibrated = {
        'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
        'MAE': mean_absolute_error(y_test, test_pred),
        'R2': r2_score(y_test, test_pred)
    }
    
    test_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, test_pred_calibrated)),
        'MAE': mean_absolute_error(y_test, test_pred_calibrated),
        'R2': r2_score(y_test, test_pred_calibrated)
    }
    
    # Calculate baseline RMSE
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE (uncalibrated): {test_metrics_uncalibrated['RMSE']:.4f}, R²: {test_metrics_uncalibrated['R2']:.4f}")
    print(f"   Test RMSE (calibrated):   {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    print(f"   Calibration improvement: {((test_metrics_uncalibrated['RMSE'] - test_metrics['RMSE']) / test_metrics_uncalibrated['RMSE'] * 100):.2f}%")
    
    return model, calibrator, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== REMOVED: BLENDING FUNCTION (NOT USED - GLOBAL MODEL ONLY) ====================
# This function is no longer used since we only use the global NO2 model
def evaluate_improved_blending(df, seasonal_models, seasonal_calibrators, test_mask, features, global_rmse):
    """Evaluate improved blending with smart fallback and adaptive logic"""
    print(f"\n{'='*80}")
    print("EVALUATING IMPROVED BLENDING")
    print(f"{'='*80}")
    
    valid_mask = ~df['NO2_target'].isna()
    test_idx = valid_mask & test_mask
    X_test = df[test_idx][features].copy()
    y_test = df[test_idx]['NO2_target'].copy()
    test_months = df[test_idx]['month'].values
    
    # Convert to numeric
    for col in features:
        if col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = pd.Categorical(X_test[col]).codes
            elif X_test[col].dtype == 'bool':
                X_test[col] = X_test[col].astype(int)
    
    X_test = X_test.select_dtypes(include=[np.number])
    features = [f for f in features if f in X_test.columns]
    
    for col in X_test.columns:
        # Ensure we get a scalar value
        col_series = X_test[col]
        if isinstance(col_series, pd.Series):
            null_count = int(col_series.isnull().sum())
        else:
            null_count = 0
        
        if null_count > 0:
            X_test[col].fillna(X_test[col].median(), inplace=True)
    
    # Predictions
    global_pred = seasonal_models['global'].predict(
        X_test, num_iteration=seasonal_models['global'].best_iteration
    )
    
    # Apply calibration
    if seasonal_calibrators['global'] is not None:
        global_pred = seasonal_calibrators['global'].predict(global_pred)
    
    # Calculate seasonal RMSEs for adaptive blending
    seasonal_rmses = {}
    for season_name in ['winter', 'post_monsoon']:
        if season_name in seasonal_models and seasonal_models[season_name] is not None:
            season_mask = None
            if season_name == 'winter':
                season_mask = df[test_idx]['month'].isin([12, 1, 2])
            elif season_name == 'post_monsoon':
                season_mask = df[test_idx]['month'].isin([10, 11])
            
            if season_mask is not None and season_mask.sum() > 0:
                season_pred = seasonal_models[season_name].predict(
                    X_test[season_mask], num_iteration=seasonal_models[season_name].best_iteration
                )
                if season_name in seasonal_calibrators and seasonal_calibrators[season_name] is not None:
                    season_pred = seasonal_calibrators[season_name].predict(season_pred)
                seasonal_rmses[season_name] = np.sqrt(mean_squared_error(y_test[season_mask], season_pred))
            else:
                seasonal_rmses[season_name] = float('inf')
        else:
            seasonal_rmses[season_name] = float('inf')
    
    # Blended predictions with improved logic
    blended_pred = []
    soft_blended_pred = []
    adaptive_blended_pred = []
    season_used = []
    
    for i, month in enumerate(test_months):
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5, 6]:
            season = 'summer'
        elif month in [7, 8, 9]:
            season = 'monsoon'
        else:
            season = 'post_monsoon'
        
        # Get seasonal prediction if available
        if season in seasonal_models and seasonal_models[season] is not None:
            seasonal_pred = seasonal_models[season].predict(
                X_test.iloc[[i]], num_iteration=seasonal_models[season].best_iteration
            )[0]
            
            # Apply calibration
            if season in seasonal_calibrators and seasonal_calibrators[season] is not None:
                seasonal_pred = seasonal_calibrators[season].predict([seasonal_pred])[0]
            
            # STEP 2: Hard blending - always use seasonal if available (winter highest priority)
            if season == 'winter':
                blended_pred.append(seasonal_pred)
                season_used.append('winter')
            else:
                blended_pred.append(seasonal_pred)
                season_used.append(season)
            
            # STEP 5: Soft blending: 70% seasonal + 30% global (exact specification)
            soft_blended_pred.append(0.7 * seasonal_pred + 0.3 * global_pred[i])
            
            # STEP 2: Adaptive blending - use seasonal only if it's better than global
            if season in seasonal_rmses and seasonal_rmses[season] < global_rmse:
                adaptive_blended_pred.append(seasonal_pred)
            else:
                adaptive_blended_pred.append(global_pred[i])
        else:
            # Fallback to global
            blended_pred.append(global_pred[i])
            soft_blended_pred.append(global_pred[i])
            adaptive_blended_pred.append(global_pred[i])
            season_used.append('global')
    
    blended_pred = np.array(blended_pred)
    soft_blended_pred = np.array(soft_blended_pred)
    adaptive_blended_pred = np.array(adaptive_blended_pred)
    
    # Evaluate all blending strategies
    blended_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, blended_pred)),
        'MAE': mean_absolute_error(y_test, blended_pred),
        'R2': r2_score(y_test, blended_pred)
    }
    
    soft_blended_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, soft_blended_pred)),
        'MAE': mean_absolute_error(y_test, soft_blended_pred),
        'R2': r2_score(y_test, soft_blended_pred)
    }
    
    adaptive_blended_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, adaptive_blended_pred)),
        'MAE': mean_absolute_error(y_test, adaptive_blended_pred),
        'R2': r2_score(y_test, adaptive_blended_pred)
    }
    
    global_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, global_pred)),
        'MAE': mean_absolute_error(y_test, global_pred),
        'R2': r2_score(y_test, global_pred)
    }
    
    # Winter-specific metrics
    winter_mask = df[test_idx]['month'].isin([12, 1, 2])
    if winter_mask.sum() > 0:
        winter_y = y_test[winter_mask]
        winter_blended = blended_pred[winter_mask]
        winter_soft = soft_blended_pred[winter_mask]
        winter_adaptive = adaptive_blended_pred[winter_mask]
        winter_global = global_pred[winter_mask]
        
        winter_blended_rmse = np.sqrt(mean_squared_error(winter_y, winter_blended))
        winter_soft_rmse = np.sqrt(mean_squared_error(winter_y, winter_soft))
        winter_adaptive_rmse = np.sqrt(mean_squared_error(winter_y, winter_adaptive))
        winter_global_rmse = np.sqrt(mean_squared_error(winter_y, winter_global))
        
        print(f"\n   Winter-Specific Performance:")
        print(f"   Global RMSE:        {winter_global_rmse:.4f}")
        print(f"   Blended RMSE:       {winter_blended_rmse:.4f} (improvement: {((winter_global_rmse - winter_blended_rmse) / winter_global_rmse * 100):.2f}%)")
        print(f"   Soft Blended RMSE:  {winter_soft_rmse:.4f} (improvement: {((winter_global_rmse - winter_soft_rmse) / winter_global_rmse * 100):.2f}%)")
        print(f"   Adaptive RMSE:      {winter_adaptive_rmse:.4f} (improvement: {((winter_global_rmse - winter_adaptive_rmse) / winter_global_rmse * 100):.2f}%)")
    
    # Post-monsoon-specific metrics
    post_monsoon_mask = df[test_idx]['month'].isin([10, 11])
    if post_monsoon_mask.sum() > 0:
        pm_y = y_test[post_monsoon_mask]
        pm_blended = blended_pred[post_monsoon_mask]
        pm_soft = soft_blended_pred[post_monsoon_mask]
        pm_adaptive = adaptive_blended_pred[post_monsoon_mask]
        pm_global = global_pred[post_monsoon_mask]
        
        pm_blended_rmse = np.sqrt(mean_squared_error(pm_y, pm_blended))
        pm_soft_rmse = np.sqrt(mean_squared_error(pm_y, pm_soft))
        pm_adaptive_rmse = np.sqrt(mean_squared_error(pm_y, pm_adaptive))
        pm_global_rmse = np.sqrt(mean_squared_error(pm_y, pm_global))
        
        print(f"\n   Post-Monsoon-Specific Performance:")
        print(f"   Global RMSE:        {pm_global_rmse:.4f}")
        print(f"   Blended RMSE:       {pm_blended_rmse:.4f} (improvement: {((pm_global_rmse - pm_blended_rmse) / pm_global_rmse * 100):.2f}%)")
        print(f"   Soft Blended RMSE:  {pm_soft_rmse:.4f} (improvement: {((pm_global_rmse - pm_soft_rmse) / pm_global_rmse * 100):.2f}%)")
        print(f"   Adaptive RMSE:      {pm_adaptive_rmse:.4f} (improvement: {((pm_global_rmse - pm_adaptive_rmse) / pm_global_rmse * 100):.2f}%)")
    
    print(f"\n   Overall Performance:")
    print(f"   Global RMSE:        {global_metrics['RMSE']:.4f}, R²: {global_metrics['R2']:.4f}")
    print(f"   Blended RMSE:       {blended_metrics['RMSE']:.4f}, R²: {blended_metrics['R2']:.4f}")
    print(f"   Soft Blended RMSE:  {soft_blended_metrics['RMSE']:.4f}, R²: {soft_blended_metrics['R2']:.4f}")
    print(f"   Adaptive RMSE:      {adaptive_blended_metrics['RMSE']:.4f}, R²: {adaptive_blended_metrics['R2']:.4f}")
    
    improvement = ((global_metrics['RMSE'] - blended_metrics['RMSE']) / global_metrics['RMSE'] * 100)
    soft_improvement = ((global_metrics['RMSE'] - soft_blended_metrics['RMSE']) / global_metrics['RMSE'] * 100)
    adaptive_improvement = ((global_metrics['RMSE'] - adaptive_blended_metrics['RMSE']) / global_metrics['RMSE'] * 100)
    print(f"   Blended Improvement: {improvement:.2f}%")
    print(f"   Soft Blended Improvement: {soft_improvement:.2f}%")
    print(f"   Adaptive Improvement: {adaptive_improvement:.2f}%")
    
    return blended_metrics, soft_blended_metrics, adaptive_blended_metrics, global_metrics, blended_pred, soft_blended_pred, adaptive_blended_pred

# ==================== MAIN ====================
print("\n3. Training models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

train_mask = (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-03-31')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

# Train O3 model first (unchanged)
o3_model, o3_train_metrics, o3_val_metrics, o3_test_metrics, o3_baseline = train_o3_model(
    df, train_mask, val_mask, test_mask
)

o3_model.save_model('models/final_o3_model.txt')
with open('models/final_o3_model.pkl', 'wb') as f:
    pickle.dump(o3_model, f)

results_summary.append({
    'Model': 'O3_target',
    'Train_RMSE': o3_train_metrics['RMSE'],
    'Train_R2': o3_train_metrics['R2'],
    'Val_RMSE': o3_val_metrics['RMSE'],
    'Val_R2': o3_val_metrics['R2'],
    'Test_RMSE': o3_test_metrics['RMSE'],
    'Test_MAE': o3_test_metrics['MAE'],
    'Test_R2': o3_test_metrics['R2'],
    'Baseline_RMSE': o3_baseline,
    'Improvement_%': ((o3_baseline - o3_test_metrics['RMSE']) / o3_baseline * 100)
})

# Train NO2 global model only
no2_model, no2_calibrator, no2_train_metrics, no2_val_metrics, no2_test_metrics, no2_baseline = train_no2_model(
    df, train_mask, val_mask, test_mask
)

# Save NO2 model
no2_model.save_model('models/final_no2_model.txt')
with open('models/final_no2_model.pkl', 'wb') as f:
    pickle.dump((no2_model, no2_calibrator), f)

results_summary.append({
    'Model': 'NO2_target',
    'Train_RMSE': no2_train_metrics['RMSE'],
    'Train_R2': no2_train_metrics['R2'],
    'Val_RMSE': no2_val_metrics['RMSE'],
    'Val_R2': no2_val_metrics['R2'],
    'Test_RMSE': no2_test_metrics['RMSE'],
    'Test_MAE': no2_test_metrics['MAE'],
    'Test_R2': no2_test_metrics['R2'],
    'Baseline_RMSE': no2_baseline,
    'Improvement_%': ((no2_baseline - no2_test_metrics['RMSE']) / no2_baseline * 100)
})

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/final_no2_o3_performance_summary.csv', index=False)

# ==================== CREATE NO2 MODEL DOCUMENTATION ====================
# Get feature importance from model
no2_features = get_no2_features(df, include_winter_features=True)
if no2_model is not None:
    feature_importance = no2_model.feature_importance(importance_type='gain')
    feature_names = no2_model.feature_name()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('results/no2_feature_importance.csv', index=False)

# 1. Create NO2_MODEL_FEATURES.txt
with open('results/NO2_MODEL_FEATURES.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("NO2 MODEL - ALL FEATURES USED FOR TRAINING\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total Features: {len(no2_features)}\n\n")
    f.write("FEATURES BY CATEGORY:\n")
    f.write("-"*80 + "\n\n")
    
    # Categorize features
    core_pollutants = [f for f in no2_features if f in ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no']]
    meteo_features = [f for f in no2_features if f in ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5', 'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']]
    lag_features = [f for f in no2_features if '_lag_' in f]
    rolling_features = [f for f in no2_features if '_rolling_mean_' in f]
    interaction_features = [f for f in no2_features if any(x in f for x in ['_ratio', '_product', '_interaction', 'blh_wind', 'blh_inversion', 'no2lag_blhlag', 'hour_weekend'])]
    time_features = [f for f in no2_features if f in ['month', 'day_of_week', 'is_weekday', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'hour', 'is_weekend']]
    season_features = [f for f in no2_features if f.startswith('is_') and f in ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']]
    winter_features = [f for f in no2_features if f in ['inversion_strength', 'inversion_strength_abs', 'is_night', 'morning_peak', 'blh_inversion', 'blh_inv', 'no2lag_blhlag', 'no2_blhlag', 'temperature_roll3']]
    post_monsoon_features = [f for f in no2_features if f in ['stubble_burning_flag', 'diwali_flag', 'festival_flag', 'low_wind_flag', 'wind_stagnation_index', 'low_blh_flag', 'low_blh']]
    season_robust = [f for f in no2_features if f in ['inv', 'ventilation', 'blh_wind', 'ventilation_rate', 'stability', 'hour_weekend_interaction', 'blh_roll3', 'blh_rolling_mean_3h']]
    blh_features = [f for f in no2_features if 'blh' in f.lower() and f not in meteo_features and f not in interaction_features and f not in season_robust]
    wind_features = [f for f in no2_features if 'wind' in f.lower() and f not in meteo_features and f not in interaction_features]
    other_features = [f for f in no2_features if f not in core_pollutants and f not in meteo_features and f not in lag_features and f not in rolling_features and f not in interaction_features and f not in time_features and f not in season_features and f not in winter_features and f not in post_monsoon_features and f not in season_robust and f not in blh_features and f not in wind_features]
    
    f.write("1. CORE POLLUTANTS:\n")
    for feat in core_pollutants:
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(core_pollutants)}\n\n")
    
    f.write("2. METEOROLOGICAL FEATURES:\n")
    for feat in meteo_features:
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(meteo_features)}\n\n")
    
    f.write("3. LAG FEATURES:\n")
    for feat in sorted(lag_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(lag_features)}\n\n")
    
    f.write("4. ROLLING MEAN FEATURES:\n")
    for feat in sorted(rolling_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(rolling_features)}\n\n")
    
    f.write("5. INTERACTION FEATURES:\n")
    for feat in sorted(interaction_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(interaction_features)}\n\n")
    
    f.write("6. TIME FEATURES:\n")
    for feat in sorted(time_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(time_features)}\n\n")
    
    f.write("7. SEASON FEATURES:\n")
    for feat in sorted(season_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(season_features)}\n\n")
    
    f.write("8. SEASON-ROBUST FEATURES (High impact for R²):\n")
    for feat in sorted(season_robust):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(season_robust)}\n\n")
    
    f.write("9. WINTER-SPECIFIC FEATURES:\n")
    for feat in sorted(winter_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(winter_features)}\n\n")
    
    f.write("10. POST-MONSOON FEATURES:\n")
    for feat in sorted(post_monsoon_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(post_monsoon_features)}\n\n")
    
    f.write("11. BLH-BASED FEATURES:\n")
    for feat in sorted(blh_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(blh_features)}\n\n")
    
    f.write("12. WIND COMPONENT FEATURES:\n")
    for feat in sorted(wind_features):
        f.write(f"   - {feat}\n")
    f.write(f"   Total: {len(wind_features)}\n\n")
    
    if other_features:
        f.write("13. OTHER FEATURES:\n")
        for feat in sorted(other_features):
            f.write(f"   - {feat}\n")
        f.write(f"   Total: {len(other_features)}\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ALL FEATURES IN MODEL ORDER:\n")
    f.write("-"*80 + "\n")
    for i, feat in enumerate(no2_features, 1):
        f.write(f"{i:3d}. {feat}\n")
    f.write("\n" + "="*80 + "\n")

# 2. Create NO2_MODEL_INFORMATION.txt
with open('results/NO2_MODEL_INFORMATION.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("NO2 MODEL - COMPLETE MODEL INFORMATION\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("MODEL ARCHITECTURE:\n")
    f.write("-"*80 + "\n")
    f.write("NO2 model uses a single global LightGBM model trained on all seasons.\n")
    f.write("The model includes:\n")
    f.write("1. Peak-weighted training (4.0x weight for high-pollution events)\n")
    f.write("2. Residual calibration using Isotonic Regression\n")
    f.write("3. Comprehensive feature engineering for all seasons\n\n")
    
    f.write("MODEL PERFORMANCE:\n")
    f.write("-"*80 + "\n")
    f.write(f"Train RMSE: {no2_train_metrics['RMSE']:.6f}\n")
    f.write(f"Train R²:   {no2_train_metrics['R2']:.6f}\n")
    f.write(f"Val RMSE:   {no2_val_metrics['RMSE']:.6f}\n")
    f.write(f"Val R²:     {no2_val_metrics['R2']:.6f}\n")
    f.write(f"Test RMSE:  {no2_test_metrics['RMSE']:.6f}\n")
    f.write(f"Test MAE:   {no2_test_metrics['MAE']:.6f}\n")
    f.write(f"Test R²:    {no2_test_metrics['R2']:.6f}\n")
    f.write(f"Baseline RMSE: {no2_baseline:.6f}\n")
    f.write(f"Improvement: {((no2_baseline - no2_test_metrics['RMSE']) / no2_baseline * 100):.2f}%\n\n")
    
    f.write("HYPERPARAMETERS:\n")
    f.write("-"*80 + "\n")
    f.write("objective: regression\n")
    f.write("metric: rmse\n")
    f.write("boosting_type: gbdt\n")
    f.write("num_leaves: 15\n")
    f.write("max_depth: 5\n")
    f.write("learning_rate: 0.03\n")
    f.write("feature_fraction: 0.7\n")
    f.write("bagging_fraction: 0.7\n")
    f.write("bagging_freq: 5\n")
    f.write("min_data_in_leaf: 50\n")
    f.write("lambda_l1: 1.0\n")
    f.write("lambda_l2: 1.0\n")
    f.write("random_state: 42\n")
    f.write("num_boost_round: 200\n")
    f.write("early_stopping_rounds: 30\n")
    f.write("sample_weight: 4.0x for NO2 > 75th percentile\n\n")
    
    f.write("DATA SPLITS:\n")
    f.write("-"*80 + "\n")
    f.write("Train Period: 2020-01-01 to 2021-12-31\n")
    f.write("Val Period:   2022-01-01 to 2022-03-31\n")
    f.write("Test Period:  2022-07-01 to 2022-12-31\n")
    train_count = train_mask.sum()
    val_count = val_mask.sum()
    test_count = test_mask.sum()
    f.write(f"Train Samples: {train_count} (approximate)\n")
    f.write(f"Val Samples:   {val_count} (approximate)\n")
    f.write(f"Test Samples:  {test_count} (approximate)\n\n")
    
    f.write("FEATURE ENGINEERING STEPS:\n")
    f.write("-"*80 + "\n")
    f.write("1. Time Features:\n")
    f.write("   - Extract: year, month, day, hour, day_of_week, day_of_year\n")
    f.write("   - Create: is_weekend, is_weekday\n")
    f.write("   - Cyclical encoding: hour_sin, hour_cos, month_sin, month_cos\n\n")
    
    f.write("2. Lag Features:\n")
    f.write("   - Create lags for: no2, pm2p5, pm10, so2, t2m_era5, wind_speed\n")
    f.write("   - Lag windows: 1h, 3h, 6h, 12h, 24h\n")
    f.write("   - BLH lags: blh_lag_1h\n\n")
    
    f.write("3. Rolling Mean Features:\n")
    f.write("   - Create rolling means for: no2, pm2p5, pm10\n")
    f.write("   - Windows: 3h, 6h, 12h, 24h\n")
    f.write("   - BLH rolling: blh_roll3 (3-hour rolling mean)\n")
    f.write("   - Temperature rolling: temperature_roll3 (3-hour rolling mean)\n\n")
    
    f.write("4. Season-Robust Features (High impact for R²):\n")
    f.write("   - inv: t2m_era5 - d2m_era5 (inversion strength)\n")
    f.write("   - ventilation: blh_era5 × wind_speed (ventilation index)\n")
    f.write("   - stability: inv × (1 / blh_era5) (stability index)\n")
    f.write("   - hour_weekend_interaction: hour × is_weekend (traffic pattern)\n\n")
    
    f.write("5. Winter-Specific Features:\n")
    f.write("   - inversion_strength: t2m_era5 - d2m_era5\n")
    f.write("   - is_night: 1 if (hour < 7 or hour > 20) else 0\n")
    f.write("   - morning_peak: 1 if hour in [7,8,9] else 0\n")
    f.write("   - blh_wind: blh_era5 × wind_speed\n")
    f.write("   - blh_inversion: blh_era5 × inversion_strength\n")
    f.write("   - no2lag_blhlag: NO2_target_lag1 × blh_era5\n\n")
    
    f.write("6. Post-Monsoon Features:\n")
    f.write("   - stubble_burning_flag: 1 if month in [10,11] else 0\n")
    f.write("   - diwali_flag: 1 for 5 days around Diwali (Oct 20-24, Nov 1-5)\n")
    f.write("   - low_wind_flag: 1 if wind_speed < 1.0 else 0\n")
    f.write("   - low_blh_flag: 1 if blh_era5 < 100 else 0\n\n")
    
    f.write("7. Interaction Features:\n")
    f.write("   - PM interactions: pm25_pm10_ratio, pm25_pm10_product\n")
    f.write("   - NO2-PM interactions: no2_pm25_ratio, no2_pm25_product\n")
    f.write("   - BLH interactions: blh_temp_interaction, blh_no2_lag1_interaction\n")
    f.write("   - Hour interactions: hour_temp_interaction\n\n")
    
    f.write("8. Wind Component Features:\n")
    f.write("   - wind_u, wind_v (u10_era5, v10_era5)\n")
    f.write("   - wind_u_abs, wind_v_abs\n")
    f.write("   - wind_uv_product\n\n")
    
    f.write("TRAINING PROCESS:\n")
    f.write("-"*80 + "\n")
    f.write("1. Peak-Weighted Training:\n")
    f.write("   - Samples with NO2_target >= 75th percentile get 4.0x weight\n")
    f.write("   - Improves prediction of high-pollution events\n\n")
    
    f.write("2. Residual Calibration:\n")
    f.write("   - Isotonic regression calibrator trained on validation set\n")
    f.write("   - Applied to test predictions to reduce bias\n")
    f.write("   - Calibrator saved with model in pickle file\n\n")
    
    if no2_model is not None and 'importance_df' in locals():
        f.write("TOP 20 MOST IMPORTANT FEATURES:\n")
        f.write("-"*80 + "\n")
        for i, row in importance_df.head(20).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:15.2f}\n")
        f.write("\n")
    
    f.write("DATA PREPROCESSING:\n")
    f.write("-"*80 + "\n")
    f.write("1. Missing values: Filled with median of training data\n")
    f.write("2. Categorical features: Encoded using pandas Categorical codes\n")
    f.write("3. Boolean features: Converted to integers (0/1)\n")
    f.write("4. Feature selection: Only numeric features passed to LightGBM\n")
    f.write("5. Duplicate removal: Features deduplicated while preserving order\n\n")
    
    f.write("MODEL FILES:\n")
    f.write("-"*80 + "\n")
    f.write("NO2 Model: models/final_no2_model.txt/.pkl\n")
    f.write("(Pickle file includes both model and calibrator)\n\n")
    
    f.write("REPLICATION INSTRUCTIONS:\n")
    f.write("-"*80 + "\n")
    f.write("1. Load the data: master_site1_final_cleaned.csv\n")
    f.write("2. Run feature engineering as described above\n")
    f.write("3. Use the same train/val/test splits\n")
    f.write("4. Load model from pickle file (includes calibrator)\n")
    f.write("5. Make predictions using the model\n")
    f.write("6. Apply calibrator to predictions to reduce bias\n\n")
    
    f.write("="*80 + "\n")

# Create final documentation
with open('results/FINAL_NO2_IMPROVEMENTS.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("FINAL NO2 MODEL - GLOBAL MODEL ONLY\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("MODEL ARCHITECTURE:\n")
    f.write("-"*80 + "\n")
    f.write("Single global LightGBM model trained on all seasons.\n")
    f.write("No seasonal specialists or blending used.\n\n")
    
    f.write("KEY IMPROVEMENTS:\n")
    f.write("-"*80 + "\n")
    f.write("1. SEASON-ROBUST FEATURES (High impact for R²):\n")
    f.write("   (A) Inversion strength: inv = t2m - d2m\n")
    f.write("   (B) Ventilation index: ventilation = blh × wind_speed\n")
    f.write("   (C) Stability index: stability = inv × (1 / blh)\n")
    f.write("   (D) Traffic proxies: hour, is_weekend, (hour × is_weekend)\n")
    f.write("   (E) Smooth BLH: blh_roll3 = blh.rolling(3).mean()\n\n")
    
    f.write("2. COMPREHENSIVE FEATURE ENGINEERING:\n")
    f.write("   - Extended lag features (1h, 3h, 6h, 12h, 24h)\n")
    f.write("   - Rolling mean features (3h, 6h, 12h, 24h)\n")
    f.write("   - PM interactions (pm25_pm10_ratio, pm25_pm10_product)\n")
    f.write("   - Wind components (u, v, abs values, products)\n")
    f.write("   - Winter-specific features (inversion_strength, is_night, morning_peak, etc.)\n")
    f.write("   - Post-monsoon features (stubble_burning_flag, diwali_flag, low_wind_flag, etc.)\n\n")
    
    f.write("3. PEAK-WEIGHTED TRAINING:\n")
    f.write("   - Weight: 4.0x for NO2_target >= 75th percentile\n")
    f.write("   - Improves prediction of high-pollution events\n\n")
    
    f.write("4. RESIDUAL CALIBRATION:\n")
    f.write("   - Isotonic regression calibrator trained on validation set\n")
    f.write("   - Applied to test predictions to reduce bias\n")
    f.write("   - Calibrator saved with model in pickle file\n\n")
    
    f.write("PERFORMANCE:\n")
    f.write("-"*80 + "\n")
    f.write(f"Train RMSE: {no2_train_metrics['RMSE']:.6f}, R²: {no2_train_metrics['R2']:.6f}\n")
    f.write(f"Val RMSE:   {no2_val_metrics['RMSE']:.6f}, R²: {no2_val_metrics['R2']:.6f}\n")
    f.write(f"Test RMSE:  {no2_test_metrics['RMSE']:.6f}, R²: {no2_test_metrics['R2']:.6f}\n")
    f.write(f"Baseline RMSE: {no2_baseline:.6f}\n")
    f.write(f"Improvement: {((no2_baseline - no2_test_metrics['RMSE']) / no2_baseline * 100):.2f}%\n\n")
    
    f.write("MODEL FILES:\n")
    f.write("-"*80 + "\n")
    f.write("O3 Model: models/final_o3_model.txt/.pkl\n")
    f.write("NO2 Model: models/final_no2_model.txt/.pkl\n")
    f.write("(Pickle files include both model and calibrator)\n\n")
    
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n" + "="*80)


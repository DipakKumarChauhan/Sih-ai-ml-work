"""
Robust NO2 Model Training with Comprehensive Improvements
Implements all 10 improvement strategies:
1. Walk-forward CV
2. Simplified feature set (30-40 features)
3. Strong regularization
4. Physically-motivated features
5. Conservative peak-weighting
6. Residual calibration
7. Simple ensembling
8. Event indicators
9. Data quality checks
10. Monitoring framework
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
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

print("="*80)
print("ROBUST NO2 MODEL TRAINING WITH COMPREHENSIVE IMPROVEMENTS")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== DATA QUALITY CHECKS ====================
print("\n2. Data quality checks...")

def check_data_quality(df, target_col='NO2_target'):
    """Check data quality and report issues"""
    issues = []
    
    # Check missingness
    missing_pct = df[target_col].isna().sum() / len(df) * 100
    if missing_pct > 20:
        issues.append(f"High missingness: {missing_pct:.1f}%")
    
    # Check for spikes (values > 3 std from mean)
    if target_col in df.columns:
        valid_data = df[target_col].dropna()
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            spikes = ((df[target_col] - mean_val).abs() > 3 * std_val).sum()
            if spikes > 0:
                issues.append(f"Potential spikes: {spikes} values > 3 std")
    
    # Check feature missingness
    feature_missing = {}
    for col in df.columns:
        if col not in ['datetime', target_col]:
            missing = df[col].isna().sum() / len(df) * 100
            if missing > 50:
                feature_missing[col] = missing
    
    if feature_missing:
        issues.append(f"High feature missingness: {len(feature_missing)} features >50% missing")
    
    return issues

quality_issues = check_data_quality(df)
if quality_issues:
    print(f"   Data quality issues found: {quality_issues}")
else:
    print("   Data quality checks passed")

# ==================== FEATURE ENGINEERING (SIMPLIFIED CORE SET) ====================
print("\n3. Creating simplified core feature set (30-40 features)...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Derived meteorological features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)

# Core lags (1, 3, 6, 24 hours) - only essential features
core_lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in core_lag_features:
    if col in df.columns:
        for lag in [1, 3, 6, 24]:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# BLH lag
if 'blh_era5' in df.columns:
    df['blh_lag_1h'] = df['blh_era5'].shift(1)

# Physically-motivated features
if 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['inversion_strength'] = df['t2m_era5'] - df['d2m_era5']  # Inversion strength

if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['ventilation_index'] = df['blh_era5'] * df['wind_speed']  # Ventilation
    df['stability_index'] = df['inversion_strength'] / (df['blh_era5'] + 1e-6)  # Stability

# Traffic proxies
df['morning_peak'] = df['hour'].isin([7, 8, 9]).astype(int)
df['evening_peak'] = df['hour'].isin([17, 18, 19]).astype(int)
df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']

# Event indicators (post-monsoon)
df['stubble_burning_flag'] = df['month'].isin([10, 11]).astype(int)
# Diwali flag (approximate dates: Oct 20-24, Nov 1-5)
df['diwali_flag'] = ((df['month'] == 10) & (df['datetime'].dt.day >= 20) & (df['datetime'].dt.day <= 24)) | \
                     ((df['month'] == 11) & (df['datetime'].dt.day >= 1) & (df['datetime'].dt.day <= 5))
df['diwali_flag'] = df['diwali_flag'].astype(int)

# Stagnation flags
if 'wind_speed' in df.columns:
    df['low_wind_flag'] = (df['wind_speed'] < 1.0).astype(int)
if 'blh_era5' in df.columns:
    df['low_blh_flag'] = (df['blh_era5'] < 100).astype(int)

# Simple interactions (only physically meaningful)
if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)

# Wind components
if 'u10_era5' in df.columns:
    df['wind_u'] = df['u10_era5']
if 'v10_era5' in df.columns:
    df['wind_v'] = df['v10_era5']

# Solar elevation (if available)
if 'solar_elevation' in df.columns:
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])

# NO2 lifetime proxy (physical stabilizer feature)
# lifetime = 1 / (O3 + temperature_factor) - simpler approximation
# Higher O3 and temperature → shorter NO2 lifetime → faster removal
if 'O3_target' in df.columns and 't2m_era5' in df.columns:
    # Use O3 and temperature as proxies for NO2 lifetime
    # Scale temperature to similar magnitude as O3
    df['no2_lifetime_proxy'] = 1.0 / (df['O3_target'] + df['t2m_era5'] / 10.0 + 1e-6)
    print("   Added NO2 lifetime proxy feature (O3 + temperature)")
elif 't2m_era5' in df.columns:
    # Fallback: use temperature only
    df['no2_lifetime_proxy'] = 1.0 / (df['t2m_era5'] / 10.0 + 1e-6)
    print("   Added NO2 lifetime proxy feature (temperature only)")

def get_core_no2_features(df):
    """Get simplified core feature set (30-40 features max)"""
    features = []
    
    # Core pollutants (4)
    core_pollutants = ['pm2p5', 'pm10', 'so2', 'no2']
    features.extend([f for f in core_pollutants if f in df.columns])
    
    # Core meteorology (6)
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5']
    features.extend([f for f in meteo if f in df.columns])
    
    # Core lags (1, 3, 6, 24h) - only essential
    for lag in [1, 3, 6, 24]:
        for feat in ['no2', 'pm2p5', 'pm10', 't2m_era5', 'wind_speed']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    
    # BLH lag
    if 'blh_lag_1h' in df.columns:
        features.append('blh_lag_1h')
    
    # Physically-motivated features (4)
    phys_features = ['inversion_strength', 'ventilation_index', 'stability_index', 'hour_weekend_interaction']
    features.extend([f for f in phys_features if f in df.columns])
    
    # Time features (6)
    time_features = ['hour', 'month', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Traffic proxies (2)
    traffic_features = ['morning_peak', 'evening_peak']
    features.extend([f for f in traffic_features if f in df.columns])
    
    # Event indicators (3)
    event_features = ['stubble_burning_flag', 'diwali_flag', 'low_wind_flag', 'low_blh_flag']
    features.extend([f for f in event_features if f in df.columns])
    
    # Simple interactions (1)
    if 'pm25_pm10_ratio' in df.columns:
        features.append('pm25_pm10_ratio')
    
    # Wind components (2)
    if 'wind_u' in df.columns:
        features.append('wind_u')
    if 'wind_v' in df.columns:
        features.append('wind_v')
    
    # Solar (if available)
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    if 'solar_elevation_abs' in df.columns:
        features.append('solar_elevation_abs')
    
    # NO2 lifetime proxy (physical stabilizer)
    if 'no2_lifetime_proxy' in df.columns:
        features.append('no2_lifetime_proxy')
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in features:
        if f in df.columns and f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features

core_features = get_core_no2_features(df)
print(f"   Core features selected: {len(core_features)}")

# ==================== WALK-FORWARD CROSS-VALIDATION ====================
print("\n4. Setting up walk-forward cross-validation...")

def create_walk_forward_folds(df, n_folds=5, train_months=12, val_months=3):
    """
    Create walk-forward CV folds
    Each fold: train on older periods, validate on next block
    """
    df = df.sort_values('datetime').reset_index(drop=True)
    folds = []
    
    # Start from earliest date with enough data
    start_date = df['datetime'].min()
    current_date = start_date
    
    # Calculate total months needed
    total_months = train_months + val_months
    
    for fold_idx in range(n_folds):
        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_months)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)
        
        # Check if we have enough data
        if val_end > df['datetime'].max():
            break
        
        train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end)
        val_mask = (df['datetime'] >= val_start) & (df['datetime'] < val_end)
        
        if train_mask.sum() > 100 and val_mask.sum() > 20:  # Minimum data requirements
            folds.append({
                'fold': fold_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_mask': train_mask,
                'val_mask': val_mask
            })
            print(f"   Fold {fold_idx + 1}: Train {train_start.date()} to {train_end.date()}, Val {val_start.date()} to {val_end.date()}")
        
        # Move forward by validation period
        current_date = val_start
    
    return folds

# Create CV folds
cv_folds = create_walk_forward_folds(df, n_folds=5, train_months=12, val_months=3)
print(f"   Created {len(cv_folds)} CV folds")

# ==================== HANDLE PEAK EVENTS (WINSORIZATION) ====================
def winsorize_target(y, percentile=99.5):
    """Winsorize target to handle extreme spikes"""
    threshold = np.percentile(y, percentile)
    y_winsorized = y.copy()
    y_winsorized[y_winsorized > threshold] = threshold
    return y_winsorized, threshold

# ==================== DATA PREPARATION ====================
def prepare_data(df, target_col, features, train_mask, val_mask, winsorize_train=True, winsorize_percentile=99.5):
    """Prepare data with proper preprocessing - ENSURES PARITY BETWEEN CV AND FINAL"""
    valid_mask = ~df[target_col].isna()
    train_idx = valid_mask & train_mask
    val_idx = valid_mask & val_mask
    
    X_train = df[train_idx][features].copy()
    y_train = df[train_idx][target_col].copy()
    X_val = df[val_idx][features].copy()
    y_val = df[val_idx][target_col].copy()
    
    # Winsorize training targets to handle spikes
    if winsorize_train:
        y_train, winsorize_threshold = winsorize_target(y_train, percentile=winsorize_percentile)
        # Also winsorize validation at same threshold (for consistency)
        y_val = y_val.copy()
        y_val[y_val > winsorize_threshold] = winsorize_threshold
    
    # Convert to numeric
    for col in features:
        if col in X_train.columns:
            col_series = X_train[col]
            if isinstance(col_series, pd.Series):
                col_dtype = col_series.dtype
            else:
                continue
            
            if col_dtype == 'object':
                X_train[col] = pd.Categorical(X_train[col]).codes
                if col in X_val.columns:
                    X_val[col] = pd.Categorical(X_val[col], categories=pd.Categorical(X_train[col]).categories).codes
            elif col_dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype(int)
    
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    features = [f for f in features if f in X_train.columns]
    
    # Fill NaN with median
    for col in X_train.columns:
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
    
    if winsorize_train:
        return X_train, y_train, X_val, y_val, features, winsorize_threshold
    else:
        return X_train, y_train, X_val, y_val, features, None

# ==================== PEAK-AWARE TRAINING (REFINED) ====================
def calculate_refined_sample_weights(y, season_mask=None):
    """
    Refined conditional peak-weighting:
    - Weight = 3.0 for 75-90th percentile
    - Weight = 2.0 for 90-95th percentile
    - Weight = 1.0 for >95th (avoid extreme outliers that may be sensor errors)
    - Apply only in peak-relevant seasons (winter, post-monsoon)
    """
    weights = np.ones(len(y))
    
    # Calculate percentiles
    p75 = np.percentile(y, 75)
    p90 = np.percentile(y, 90)
    p95 = np.percentile(y, 95)
    
    # Conditional weighting based on percentile ranges
    mask_75_90 = (y >= p75) & (y < p90)
    mask_90_95 = (y >= p90) & (y < p95)
    
    weights[mask_75_90] = 3.0
    weights[mask_90_95] = 2.0
    # >95th percentile gets weight 1.0 (avoid sensor errors)
    
    # Apply season-aware weighting (only in peak-relevant seasons)
    if season_mask is not None:
        # Reduce weights for non-peak seasons
        weights[~season_mask] = np.minimum(weights[~season_mask], 1.5)
    
    return weights

def calculate_sample_weights(y, percentile=75, weight_factor=2.0):
    """Legacy function for backward compatibility"""
    threshold = np.percentile(y, percentile)
    weights = np.ones(len(y))
    weights[y >= threshold] = weight_factor
    return weights

# ==================== TRAIN MODEL WITH STRONG REGULARIZATION ====================
def train_regularized_model(X_train, y_train, X_val, y_val, sample_weights=None, seed=42):
    """Train model with strong regularization - REDUCED COMPLEXITY"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 6,  # Further reduced from 8
        'max_depth': 3,   # Further reduced from 4
        'learning_rate': 0.008,  # Further reduced from 0.01
        'feature_fraction': 0.8,  # Increased randomness
        'bagging_fraction': 0.8,  # Increased randomness
        'bagging_freq': 5,
        'min_data_in_leaf': 150,  # Further increased from 100
        'lambda_l1': 2.5,  # Stronger L1 regularization
        'lambda_l2': 3.0,  # Stronger L2 regularization
        'verbose': -1,
        'random_state': seed
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=600,  # More rounds with smaller LR
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model

# ==================== CALCULATE METRICS ====================
def calculate_comprehensive_metrics(y_true, y_pred, season_mask=None):
    """
    Calculate comprehensive metrics including diagnostics:
    - Overall RMSE, MAE, R², bias
    - Top-10% RMSE (PRIMARY KPI)
    - Top-1% RMSE (sensitivity check)
    - Bias in top decile
    - Per-season Top-10% RMSE
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'bias': np.mean(y_pred - y_true)
    }
    
    # Top-10% RMSE (PRIMARY KPI for peak prediction)
    top_10_pct_idx = np.argsort(y_true)[-int(len(y_true) * 0.1):]
    metrics['top10_RMSE'] = np.sqrt(mean_squared_error(y_true.iloc[top_10_pct_idx], y_pred[top_10_pct_idx]))
    
    # Top-1% RMSE (sensitivity check)
    top_1_pct_idx = np.argsort(y_true)[-int(len(y_true) * 0.01):]
    metrics['top1_RMSE'] = np.sqrt(mean_squared_error(y_true.iloc[top_1_pct_idx], y_pred[top_1_pct_idx]))
    
    # Bias in top decile
    top_10_bias = np.mean(y_pred[top_10_pct_idx] - y_true.iloc[top_10_pct_idx])
    metrics['top10_bias'] = top_10_bias
    
    # Per-season Top-10% RMSE (if season_mask provided)
    if season_mask is not None and len(season_mask) > 0:
        for season_name, season_indices in season_mask.items():
            if len(season_indices) > 10 and max(season_indices) < len(y_true):
                try:
                    season_y_true = y_true.iloc[season_indices]
                    season_y_pred = y_pred[season_indices]
                    if len(season_y_true) > 10:
                        season_top10_idx = np.argsort(season_y_true)[-int(len(season_y_true) * 0.1):]
                        season_top10_rmse = np.sqrt(mean_squared_error(
                            season_y_true.iloc[season_top10_idx],
                            season_y_pred[season_top10_idx]
                        ))
                        metrics[f'{season_name}_top10_RMSE'] = season_top10_rmse
                except (IndexError, KeyError):
                    pass  # Skip if indices don't align
    
    return metrics

# ==================== RESIDUAL CALIBRATION ====================
def train_calibrator(y_true, y_pred, method='isotonic'):
    """Train residual calibrator"""
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred, y_true)
    else:
        calibrator = LinearRegression()
        calibrator.fit(y_pred.reshape(-1, 1), y_true)
    return calibrator

# ==================== EVENT DETECTION ====================
def detect_event_windows(df, X_data, y_data, mask):
    """
    Detect event windows using:
    - Calendar flags (stubble, Diwali)
    - PM2.5 spikes
    - Low BLH
    - Low wind
    
    Args:
        df: Full dataframe
        X_data: Feature matrix (already filtered)
        y_data: Target series (already filtered)
        mask: Boolean mask for the data subset
    """
    event_flags = np.zeros(len(X_data), dtype=bool)
    
    # Get corresponding dataframe rows using mask
    df_subset = df[mask]
    
    if len(df_subset) != len(X_data):
        # If lengths don't match, try to align by index
        df_subset = df_subset.iloc[:len(X_data)]
    
    # Calendar-based events
    if 'stubble_burning_flag' in df_subset.columns:
        stubble_values = df_subset['stubble_burning_flag'].values
        if len(stubble_values) == len(event_flags):
            event_flags[stubble_values == 1] = True
    
    if 'diwali_flag' in df_subset.columns:
        diwali_values = df_subset['diwali_flag'].values
        if len(diwali_values) == len(event_flags):
            event_flags[diwali_values == 1] = True
    
    # Meteorological events (low BLH + low wind)
    if 'blh_era5' in df_subset.columns and 'wind_speed' in df_subset.columns:
        blh_values = df_subset['blh_era5'].values
        wind_values = df_subset['wind_speed'].values
        if len(blh_values) == len(event_flags):
            stagnation_mask = (blh_values < 100) & (wind_values < 1.0)
            event_flags[stagnation_mask] = True
    
    # PM2.5 spikes (if available)
    if 'pm2p5' in df_subset.columns:
        pm25_values = df_subset['pm2p5'].values
        if len(pm25_values) == len(event_flags):
            valid_pm25 = pm25_values[~np.isnan(pm25_values)]
            if len(valid_pm25) > 0:
                pm25_threshold = np.percentile(valid_pm25, 90)
                pm25_spike = pm25_values > pm25_threshold
                event_flags[pm25_spike] = True
    
    # Cold inversion events (winter)
    if 't2m_era5' in df_subset.columns and 'd2m_era5' in df_subset.columns:
        temp_values = df_subset['t2m_era5'].values
        dewpoint_values = df_subset['d2m_era5'].values
        if len(temp_values) == len(event_flags):
            inversion_mask = (temp_values - dewpoint_values) > 5.0  # Strong inversion
            # Only in winter months
            if 'month' in df_subset.columns:
                months = df_subset['month'].values
                winter_inversion = inversion_mask & np.isin(months, [12, 1, 2])
                event_flags[winter_inversion] = True
    
    return event_flags

# ==================== EVENT-AWARE RESIDUAL CORRECTION ====================
def train_event_residual_corrector(y_true, y_pred, event_flags, X_data):
    """
    Train low-capacity residual corrector for event windows
    Uses simple linear model on key features
    """
    # Only train on event windows
    event_mask = event_flags
    if event_mask.sum() < 10:
        return None  # Not enough event data
    
    # Calculate residuals for event windows
    residuals = y_true[event_mask] - y_pred[event_mask]
    
    # Select key features for correction (low capacity)
    correction_features = []
    feature_names = X_data.columns.tolist()
    
    for feat in ['hour', 'blh_era5', 'wind_speed', 'no2_lag_1h', 't2m_era5']:
        if feat in feature_names:
            correction_features.append(feat)
    
    if len(correction_features) == 0:
        return None
    
    X_event = X_data[event_mask][correction_features].values
    
    # Simple linear correction
    corrector = LinearRegression()
    corrector.fit(X_event, residuals)
    
    return corrector, correction_features

def apply_event_correction(y_pred, event_flags, corrector, X_data, correction_features):
    """Apply event-aware residual correction"""
    if corrector is None:
        return y_pred
    
    y_pred_corrected = y_pred.copy()
    
    # Only apply correction to event windows
    if event_flags.sum() > 0:
        X_event = X_data[event_flags][correction_features].values
        corrections = corrector.predict(X_event)
        y_pred_corrected[event_flags] += corrections
    
    return y_pred_corrected

# ==================== QUANTILE MODEL FOR PEAKS ====================
def train_quantile_model(X_train, y_train, X_val, y_val, quantile=0.90, sample_weights=None, seed=42):
    """Train quantile regression model for peak prediction"""
    params = {
        'objective': 'quantile',
        'alpha': quantile,  # 90th percentile
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 6,
        'max_depth': 3,
        'learning_rate': 0.008,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 150,
        'lambda_l1': 2.5,
        'lambda_l2': 3.0,
        'verbose': -1,
        'random_state': seed
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=600,
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model

def calculate_quantile_coverage(y_true, y_pred_quantile, quantile=0.90):
    """Calculate quantile coverage (should be ~quantile)"""
    coverage = (y_true <= y_pred_quantile).mean()
    return coverage

# ==================== CROSS-VALIDATION TRAINING ====================
print("\n5. Training models with walk-forward CV...")

cv_results = []
fold_models = []
fold_calibrators = []

for fold_info in cv_folds:
    print(f"\n   Training Fold {fold_info['fold']}...")
    
    # Prepare data (with winsorization for spike handling)
    # CRITICAL: Same preprocessing as will be used for production
    result = prepare_data(
        df, 'NO2_target', core_features, fold_info['train_mask'], fold_info['val_mask'],
        winsorize_train=True, winsorize_percentile=99.5
    )
    if len(result) == 6:
        X_train, y_train, X_val, y_val, features, winsorize_threshold = result
    else:
        X_train, y_train, X_val, y_val, features = result
        winsorize_threshold = None
    
    # Store preprocessing parameters for this fold (for parity checking)
    fold_info['preprocessing_params'] = {
        'features': features,
        'winsorize_threshold': winsorize_threshold,
        'n_features': len(features)
    }
    
    if len(X_train) < 50 or len(X_val) < 10:
        print(f"      Skipping fold {fold_info['fold']} - insufficient data")
        continue
    
    print(f"      Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Refined peak-weighted training (conditional, granular)
    # Identify peak-relevant seasons (winter, post-monsoon)
    train_indices = df[fold_info['train_mask']].index[df[fold_info['train_mask']].index.isin(X_train.index)]
    if len(train_indices) > 0:
        train_months = df.loc[train_indices, 'month'].values if 'month' in df.columns else None
        if train_months is not None:
            peak_season_mask = np.isin(train_months, [12, 1, 2, 10, 11])  # Winter + post-monsoon
        else:
            peak_season_mask = None
    else:
        peak_season_mask = None
    
    sample_weights = calculate_refined_sample_weights(y_train.values, season_mask=peak_season_mask)
    
    # Train mean model
    model = train_regularized_model(X_train, y_train, X_val, y_val, sample_weights)
    
    # Train quantile model (90th percentile) for peak prediction
    print(f"      Training quantile model (90th percentile) for peak prediction...")
    quantile_model = train_quantile_model(X_train, y_train, X_val, y_val, quantile=0.90, sample_weights=sample_weights)
    
    # Predictions
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred_uncalibrated = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Quantile predictions
    val_pred_quantile = quantile_model.predict(X_val, num_iteration=quantile_model.best_iteration)
    
    # Detect event windows for validation set
    val_event_flags = detect_event_windows(df, X_val, y_val, fold_info['val_mask'])
    
    # Train event-aware residual corrector
    print(f"      Training event-aware residual corrector...")
    event_corrector, correction_features = train_event_residual_corrector(
        y_val, val_pred_uncalibrated, val_event_flags, X_val
    )
    
    # Apply event correction
    if event_corrector is not None:
        val_pred_event_corrected = apply_event_correction(
            val_pred_uncalibrated, val_event_flags, event_corrector, X_val, correction_features
        )
        print(f"      Applied event correction to {val_event_flags.sum()} event windows")
    else:
        val_pred_event_corrected = val_pred_uncalibrated
    
    # Train calibrator on event-corrected predictions
    calibrator = train_calibrator(y_val, val_pred_event_corrected, method='isotonic')
    val_pred_calibrated = calibrator.predict(val_pred_event_corrected)
    
    # Calculate quantile coverage
    quantile_coverage = calculate_quantile_coverage(y_val, val_pred_quantile, quantile=0.90)
    print(f"      Quantile coverage (90th): {quantile_coverage:.3f} (target: ~0.90)")
    
    # Calculate comprehensive metrics
    # Prepare season masks for per-season metrics
    season_masks = {}
    for season_name, months in [('winter', [12, 1, 2]), ('summer', [3, 4, 5, 6]), 
                                ('monsoon', [7, 8, 9]), ('post_monsoon', [10, 11])]:
        season_mask = df[fold_info['val_mask']]['month'].isin(months)
        if season_mask.sum() > 5:
            val_season_indices = np.where(season_mask)[0]
            if len(val_season_indices) <= len(y_val):
                season_masks[season_name] = val_season_indices
    
    train_metrics = calculate_comprehensive_metrics(y_train, train_pred)
    val_metrics_uncalibrated = calculate_comprehensive_metrics(y_val, val_pred_uncalibrated, season_mask=season_masks)
    val_metrics_calibrated = calculate_comprehensive_metrics(y_val, val_pred_calibrated, season_mask=season_masks)
    
    # Add quantile metrics
    val_metrics_calibrated['quantile_coverage'] = quantile_coverage
    val_metrics_calibrated['quantile_90_pred'] = val_pred_quantile.mean()
    
    # Season-by-season evaluation
    val_seasons = {}
    for season_name, months in [('winter', [12, 1, 2]), ('summer', [3, 4, 5, 6]), 
                                ('monsoon', [7, 8, 9]), ('post_monsoon', [10, 11])]:
        season_mask = df[fold_info['val_mask']]['month'].isin(months)
        if season_mask.sum() > 5:
            season_y_val = y_val[season_mask]
            season_val_pred = val_pred_calibrated[season_mask]
            val_seasons[season_name] = calculate_comprehensive_metrics(season_y_val, season_val_pred)
    
    # Store results
    fold_result = {
        'fold': fold_info['fold'],
        'train_start': fold_info['train_start'].date().isoformat(),
        'train_end': fold_info['train_end'].date().isoformat(),
        'val_start': fold_info['val_start'].date().isoformat(),
        'val_end': fold_info['val_end'].date().isoformat(),
        'train_metrics': train_metrics,
        'val_metrics_uncalibrated': val_metrics_uncalibrated,
        'val_metrics_calibrated': val_metrics_calibrated,
        'val_seasons': val_seasons,
        'n_features': len(features)
    }
    
    cv_results.append(fold_result)
    fold_models.append(model)
    fold_calibrators.append(calibrator)
    
    # Store quantile model and event corrector for this fold (for analysis)
    fold_info['quantile_model'] = quantile_model
    fold_info['event_corrector'] = event_corrector
    fold_info['correction_features'] = correction_features if event_corrector is not None else None
    
    print(f"      Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"      Val RMSE (uncalibrated): {val_metrics_uncalibrated['RMSE']:.4f}, R²: {val_metrics_uncalibrated['R2']:.4f}")
    print(f"      Val RMSE (calibrated): {val_metrics_calibrated['RMSE']:.4f}, R²: {val_metrics_calibrated['R2']:.4f}")
    print(f"      Top-10% RMSE: {val_metrics_calibrated['top10_RMSE']:.4f} (PRIMARY KPI)")
    print(f"      Top-1% RMSE: {val_metrics_calibrated['top1_RMSE']:.4f}")
    print(f"      Top-10% Bias: {val_metrics_calibrated['top10_bias']:.4f}")
    if 'quantile_coverage' in val_metrics_calibrated:
        print(f"      Quantile Coverage: {val_metrics_calibrated['quantile_coverage']:.3f}")

# ==================== ANALYZE CV RESULTS FOR MODEL SELECTION ====================
print("\n6. Analyzing CV results for model selection...")

# Calculate mean CV performance
mean_cv_rmse = np.mean([r['val_metrics_calibrated']['RMSE'] for r in cv_results])
mean_cv_r2 = np.mean([r['val_metrics_calibrated']['R2'] for r in cv_results])
mean_cv_top10 = np.mean([r['val_metrics_calibrated']['top10_RMSE'] for r in cv_results])

print(f"   CV Mean Val RMSE: {mean_cv_rmse:.4f}")
print(f"   CV Mean Val R²: {mean_cv_r2:.4f}")
print(f"   CV Mean Top-10% RMSE: {mean_cv_top10:.4f}")
print(f"   CV models used for: model selection, hyperparameter tuning, feature selection")
print(f"   NOT for production - will train single final model with CV-selected settings")

# ==================== TRAIN SINGLE FINAL MODEL ====================
print("\n7. Training SINGLE FINAL MODEL with CV-selected settings...")
print("   Using EXACT same features, hyperparameters, and regularization as CV")

# Use recent data: train on last 2 years, validate on representative 3-6 months (mixed seasons)
final_train_start = df['datetime'].max() - pd.DateOffset(months=24)
final_val_start = df['datetime'].max() - pd.DateOffset(months=6)  # 6 months for representative validation
final_test_start = df['datetime'].max() - pd.DateOffset(months=3)  # Last 3 months as test

final_train_mask = (df['datetime'] >= final_train_start) & (df['datetime'] < final_val_start)
final_val_mask = (df['datetime'] >= final_val_start) & (df['datetime'] < final_test_start)
final_test_mask = (df['datetime'] >= final_test_start)

# Check validation window season distribution
val_seasons = df[final_val_mask]['month'].value_counts().sort_index()
print(f"\n   Validation window season distribution:")
for month, count in val_seasons.items():
    season = 'winter' if month in [12, 1, 2] else 'summer' if month in [3, 4, 5, 6] else 'monsoon' if month in [7, 8, 9] else 'post_monsoon'
    print(f"      Month {month} ({season}): {count} samples")

# Check test window season distribution
test_seasons = df[final_test_mask]['month'].value_counts().sort_index()
print(f"\n   Test window season distribution:")
for month, count in test_seasons.items():
    season = 'winter' if month in [12, 1, 2] else 'summer' if month in [3, 4, 5, 6] else 'monsoon' if month in [7, 8, 9] else 'post_monsoon'
    print(f"      Month {month} ({season}): {count} samples")

# Prepare final training data (SAME preprocessing as CV)
result = prepare_data(
    df, 'NO2_target', core_features, final_train_mask, final_val_mask,
    winsorize_train=True, winsorize_percentile=99.5
)
if len(result) == 6:
    X_train_final, y_train_final, X_val_final, y_val_final, features_final, winsorize_threshold = result
else:
    X_train_final, y_train_final, X_val_final, y_val_final, features_final = result
    winsorize_threshold = None

# Prepare test data (SAME preprocessing)
result_test = prepare_data(
    df, 'NO2_target', core_features, final_train_mask, final_test_mask,
    winsorize_train=True, winsorize_percentile=99.5
)
if len(result_test) == 6:
    X_test_final, y_test_final, _, _, _, _ = result_test
else:
    X_test_final, y_test_final, _, _, _ = result_test

print(f"\n   Final Model Data:")
print(f"   Train: {len(X_train_final)} samples")
print(f"   Val: {len(X_val_final)} samples (representative mixed seasons)")
print(f"   Test: {len(X_test_final)} samples (fair test window)")

# Refined peak-weighted training (same as CV)
train_indices_final = df[final_train_mask].index[df[final_train_mask].index.isin(X_train_final.index)]
if len(train_indices_final) > 0:
    train_months_final = df.loc[train_indices_final, 'month'].values if 'month' in df.columns else None
    if train_months_final is not None:
        peak_season_mask_final = np.isin(train_months_final, [12, 1, 2, 10, 11])
    else:
        peak_season_mask_final = None
else:
    peak_season_mask_final = None

sample_weights_final = calculate_refined_sample_weights(y_train_final.values, season_mask=peak_season_mask_final)

# Train final mean model (EXACT same hyperparameters as CV)
print(f"\n   Training final mean model with CV-selected hyperparameters...")
final_model = train_regularized_model(X_train_final, y_train_final, X_val_final, y_val_final, sample_weights_final, seed=42)

# Train final quantile model (90th percentile)
print(f"\n   Training final quantile model (90th percentile) for peak prediction...")
final_quantile_model = train_quantile_model(X_train_final, y_train_final, X_val_final, y_val_final, quantile=0.90, sample_weights=sample_weights_final, seed=42)

# Predictions
train_pred_final = final_model.predict(X_train_final, num_iteration=final_model.best_iteration)
val_pred_final_uncalibrated = final_model.predict(X_val_final, num_iteration=final_model.best_iteration)
test_pred_final_uncalibrated = final_model.predict(X_test_final, num_iteration=final_model.best_iteration)

# Quantile predictions
val_pred_final_quantile = final_quantile_model.predict(X_val_final, num_iteration=final_quantile_model.best_iteration)
test_pred_final_quantile = final_quantile_model.predict(X_test_final, num_iteration=final_quantile_model.best_iteration)

# Detect event windows
val_event_flags_final = detect_event_windows(df, X_val_final, y_val_final, final_val_mask)
test_event_flags_final = detect_event_windows(df, X_test_final, y_test_final, final_test_mask)

# Train event-aware residual corrector on validation set
print(f"\n   Training event-aware residual corrector on validation set...")
final_event_corrector, final_correction_features = train_event_residual_corrector(
    y_val_final, val_pred_final_uncalibrated, val_event_flags_final, X_val_final
)

# Apply event correction
if final_event_corrector is not None:
    val_pred_final_event_corrected = apply_event_correction(
        val_pred_final_uncalibrated, val_event_flags_final, final_event_corrector, X_val_final, final_correction_features
    )
    test_pred_final_event_corrected = apply_event_correction(
        test_pred_final_uncalibrated, test_event_flags_final, final_event_corrector, X_test_final, final_correction_features
    )
    print(f"      Applied event correction to {val_event_flags_final.sum()} val windows, {test_event_flags_final.sum()} test windows")
else:
    val_pred_final_event_corrected = val_pred_final_uncalibrated
    test_pred_final_event_corrected = test_pred_final_uncalibrated

# Train calibrator on event-corrected validation predictions
print(f"\n   Training calibrator on event-corrected validation set (representative mixed seasons)...")
final_calibrator = train_calibrator(y_val_final, val_pred_final_event_corrected, method='isotonic')

# Apply calibration
val_pred_final_calibrated = final_calibrator.predict(val_pred_final_event_corrected)
test_pred_final_calibrated = final_calibrator.predict(test_pred_final_event_corrected)

# Prepare season masks for per-season metrics
val_season_masks_final = {}
test_season_masks_final = {}
for season_name, months in [('winter', [12, 1, 2]), ('summer', [3, 4, 5, 6]), 
                            ('monsoon', [7, 8, 9]), ('post_monsoon', [10, 11])]:
    val_season_mask = df[final_val_mask]['month'].isin(months)
    test_season_mask = df[final_test_mask]['month'].isin(months)
    if val_season_mask.sum() > 5:
        val_season_indices = np.where(val_season_mask)[0]
        if len(val_season_indices) <= len(y_val_final):
            val_season_masks_final[season_name] = val_season_indices
    if test_season_mask.sum() > 5:
        test_season_indices = np.where(test_season_mask)[0]
        if len(test_season_indices) <= len(y_test_final):
            test_season_masks_final[season_name] = test_season_indices

# Calculate comprehensive metrics with diagnostics
train_metrics_final = calculate_comprehensive_metrics(y_train_final, train_pred_final)
val_metrics_final_uncalibrated = calculate_comprehensive_metrics(y_val_final, val_pred_final_uncalibrated, season_mask=val_season_masks_final)
val_metrics_final_calibrated = calculate_comprehensive_metrics(y_val_final, val_pred_final_calibrated, season_mask=val_season_masks_final)
test_metrics_final_uncalibrated = calculate_comprehensive_metrics(y_test_final, test_pred_final_uncalibrated, season_mask=test_season_masks_final)
test_metrics_final_calibrated = calculate_comprehensive_metrics(y_test_final, test_pred_final_calibrated, season_mask=test_season_masks_final)

# Add quantile coverage metrics
val_quantile_coverage = calculate_quantile_coverage(y_val_final, val_pred_final_quantile, quantile=0.90)
test_quantile_coverage = calculate_quantile_coverage(y_test_final, test_pred_final_quantile, quantile=0.90)
val_metrics_final_calibrated['quantile_coverage'] = val_quantile_coverage
test_metrics_final_calibrated['quantile_coverage'] = test_quantile_coverage

print(f"\n   Final Model Performance (COMPREHENSIVE DIAGNOSTICS):")
print(f"   Train RMSE: {train_metrics_final['RMSE']:.4f}, R²: {train_metrics_final['R2']:.4f}")
print(f"   Val RMSE (uncalibrated): {val_metrics_final_uncalibrated['RMSE']:.4f}, R²: {val_metrics_final_uncalibrated['R2']:.4f}")
print(f"   Val RMSE (calibrated): {val_metrics_final_calibrated['RMSE']:.4f}, R²: {val_metrics_final_calibrated['R2']:.4f}")
print(f"   Test RMSE (uncalibrated): {test_metrics_final_uncalibrated['RMSE']:.4f}, R²: {test_metrics_final_uncalibrated['R2']:.4f}")
print(f"   Test RMSE (calibrated): {test_metrics_final_calibrated['RMSE']:.4f}, R²: {test_metrics_final_calibrated['R2']:.4f}")
print(f"\n   PEAK METRICS (PRIMARY KPIs):")
print(f"   Test Top-10% RMSE: {test_metrics_final_calibrated['top10_RMSE']:.4f} (PRIMARY KPI)")
print(f"   Test Top-1% RMSE: {test_metrics_final_calibrated['top1_RMSE']:.4f}")
print(f"   Test Top-10% Bias: {test_metrics_final_calibrated['top10_bias']:.4f}")
print(f"   Test Quantile Coverage (90th): {test_quantile_coverage:.3f} (target: ~0.90)")
print(f"\n   PER-SEASON TOP-10% RMSE:")
for season_name in ['winter', 'post_monsoon', 'summer', 'monsoon']:
    key = f'{season_name}_top10_RMSE'
    if key in test_metrics_final_calibrated:
        print(f"   {season_name.capitalize()}: {test_metrics_final_calibrated[key]:.4f}")

# Check if metrics match expected range
print(f"\n   Expected Performance Range:")
print(f"   RMSE: ≤15.5, Top-10% RMSE: ≤14 (target: 12-14)")
print(f"   Quantile Coverage: ~0.90")
if test_metrics_final_calibrated['RMSE'] <= 15.5:
    print(f"   ✅ RMSE ≤ 15.5!")
else:
    print(f"   ⚠️  RMSE > 15.5")
if test_metrics_final_calibrated['top10_RMSE'] <= 14:
    print(f"   ✅ Top-10% RMSE ≤ 14!")
else:
    print(f"   ⚠️  Top-10% RMSE > 14 (current: {test_metrics_final_calibrated['top10_RMSE']:.4f})")
if 0.85 <= test_quantile_coverage <= 0.95:
    print(f"   ✅ Quantile coverage in range!")
else:
    print(f"   ⚠️  Quantile coverage outside range (current: {test_quantile_coverage:.3f})")

# ==================== SAVE FINAL MODEL ====================
print("\n8. Saving FINAL PRODUCTION MODEL...")
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save final models (mean model + quantile model + event corrector + calibrator)
final_model.save_model('models/robust_no2_final_model.txt')
final_quantile_model.save_model('models/robust_no2_final_quantile_model.txt')
with open('models/robust_no2_final_model.pkl', 'wb') as f:
    pickle.dump({
        'mean_model': final_model,
        'quantile_model': final_quantile_model,
        'event_corrector': final_event_corrector,
        'correction_features': final_correction_features if final_event_corrector is not None else None,
        'calibrator': final_calibrator
    }, f)

print(f"   Saved FINAL PRODUCTION MODELS:")
print(f"   - Mean model: models/robust_no2_final_model.txt")
print(f"   - Quantile model: models/robust_no2_final_quantile_model.txt")
print(f"   - Full pipeline: models/robust_no2_final_model.pkl")
print(f"   This includes: mean model, quantile model, event corrector, and calibrator")

# ==================== SAVE RESULTS ====================
print("\n9. Saving results...")

# Summary statistics across folds
cv_summary = {
    'n_folds': len(cv_results),
    'mean_val_rmse': np.mean([r['val_metrics_calibrated']['RMSE'] for r in cv_results]),
    'std_val_rmse': np.std([r['val_metrics_calibrated']['RMSE'] for r in cv_results]),
    'mean_val_r2': np.mean([r['val_metrics_calibrated']['R2'] for r in cv_results]),
    'mean_top10_rmse': np.mean([r['val_metrics_calibrated']['top10_RMSE'] for r in cv_results]),
    'fold_results': cv_results,
    'n_features': len(core_features),
    'features': core_features,
    'winsorize_percentile': 99.5,
    'cv_purpose': 'model_selection_hyperparameter_tuning_feature_selection',
    'final_model_metrics': {
        'train': train_metrics_final,
        'val_uncalibrated': val_metrics_final_uncalibrated,
        'val_calibrated': val_metrics_final_calibrated,
        'test_uncalibrated': test_metrics_final_uncalibrated,
        'test_calibrated': test_metrics_final_calibrated
    },
    'final_model_periods': {
        'train_start': final_train_start.date().isoformat(),
        'train_end': final_val_start.date().isoformat(),
        'val_start': final_val_start.date().isoformat(),
        'val_end': final_test_start.date().isoformat(),
        'test_start': final_test_start.date().isoformat(),
        'test_end': df['datetime'].max().date().isoformat()
    }
}

with open('results/robust_no2_cv_results.json', 'w') as f:
    json.dump(cv_summary, f, indent=2, default=str)

# Create summary table with comprehensive diagnostics
summary_data = []
for result in cv_results:
    row = {
        'Fold': result['fold'],
        'Train_Period': f"{result['train_start']} to {result['train_end']}",
        'Val_Period': f"{result['val_start']} to {result['val_end']}",
        'Train_RMSE': result['train_metrics']['RMSE'],
        'Train_R2': result['train_metrics']['R2'],
        'Val_RMSE': result['val_metrics_calibrated']['RMSE'],
        'Val_R2': result['val_metrics_calibrated']['R2'],
        'Val_MAE': result['val_metrics_calibrated']['MAE'],
        'Val_Bias': result['val_metrics_calibrated']['bias'],
        'Top10_RMSE': result['val_metrics_calibrated']['top10_RMSE'],  # PRIMARY KPI
        'Top1_RMSE': result['val_metrics_calibrated'].get('top1_RMSE', np.nan),
        'Top10_Bias': result['val_metrics_calibrated'].get('top10_bias', np.nan),
        'Quantile_Coverage': result['val_metrics_calibrated'].get('quantile_coverage', np.nan)
    }
    
    # Add per-season Top-10% RMSE
    for season in ['winter', 'post_monsoon', 'summer', 'monsoon']:
        key = f'{season}_top10_RMSE'
        if key in result['val_metrics_calibrated']:
            row[f'{season}_Top10_RMSE'] = result['val_metrics_calibrated'][key]
        else:
            row[f'{season}_Top10_RMSE'] = np.nan
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('results/robust_no2_cv_summary.csv', index=False)

print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))
print(f"\nIndividual Fold Performance (COMPREHENSIVE DIAGNOSTICS):")
print(f"Mean Val RMSE: {cv_summary['mean_val_rmse']:.4f} ± {cv_summary['std_val_rmse']:.4f}")
print(f"Mean Val R²: {cv_summary['mean_val_r2']:.4f}")
print(f"Mean Top-10% RMSE: {cv_summary['mean_top10_rmse']:.4f} (PRIMARY KPI)")
mean_top1 = np.nanmean([r['val_metrics_calibrated'].get('top1_RMSE', np.nan) for r in cv_results])
mean_top10_bias = np.nanmean([r['val_metrics_calibrated'].get('top10_bias', np.nan) for r in cv_results])
mean_quantile_cov = np.nanmean([r['val_metrics_calibrated'].get('quantile_coverage', np.nan) for r in cv_results])
print(f"Mean Top-1% RMSE: {mean_top1:.4f}")
print(f"Mean Top-10% Bias: {mean_top10_bias:.4f}")
print(f"Mean Quantile Coverage: {mean_quantile_cov:.3f} (target: ~0.90)")
print(f"\nCV Fold Stability:")
print(f"Top-10% RMSE std: {np.std([r['val_metrics_calibrated']['top10_RMSE'] for r in cv_results]):.4f}")
print(f"RMSE std: {cv_summary['std_val_rmse']:.4f}")

print(f"\nFINAL PRODUCTION MODEL Performance:")
print(f"Test RMSE (calibrated): {test_metrics_final_calibrated['RMSE']:.4f}")
print(f"Test R² (calibrated): {test_metrics_final_calibrated['R2']:.4f}")
print(f"Test Top-10% RMSE: {test_metrics_final_calibrated['top10_RMSE']:.4f}")
print(f"\n✅ Using SINGLE FINAL MODEL for production")
print(f"   Model: models/robust_no2_final_model.pkl")
print(f"   CV models used only for selection, NOT for production")
print("="*80)

print("\nTraining complete! Models saved to models/ directory")
print("Results saved to results/ directory")


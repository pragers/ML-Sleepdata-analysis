import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def safe_ratio(numerator, denominator, default=0.5):
    result = pd.Series(default, index=numerator.index)
    valid = denominator != 0
    valid &= numerator.notna() & denominator.notna()
    result[valid] = numerator[valid] / denominator[valid]
    return result

def SleepScoreHeuristic(data):
    # Rolling averages to get personalized baselines
    average_deep_fraction = data['DeepFraction'].rolling(window=30, min_periods=1).mean()
    average_REM_fraction = data['REMFraction'].rolling(window=30, min_periods=1).mean()
    SleepTimeScore = ((data['total']/data['SleepGoal'])**2).clip(0,1.1)
    # Boost score scaling to allow full 100s and wider spread
    deep_ratio = safe_ratio(data['DeepFraction'], average_deep_fraction)
    rem_ratio = safe_ratio(data['REMFraction'], average_REM_fraction)
    sleep_architecture_score = ((deep_ratio * 0.6 + rem_ratio * 0.4)**2 * 100).clip(0, 100)

    # Relative indexes are already normalized; scale them more
    HRV_score = data['HRV_relative_index']
    RHR_score = data['average_sleep_rhr_relative_index']
    VitalScore = (HRV_score*RHR_score)*100
    RespRate_score = (data['RespRate_relative_index']**1.5 * 100).clip(0, 100)

    # Improve variability in consistency scoring
    totaldif = data['TotalDif'].fillna(1800)
    consistency_score = ((1 - ((totaldif - 1200) / 9000))*100).clip(0, 100)

    # Final weighted score with more effect from variation
    sleep_score = SleepTimeScore*(
        (sleep_architecture_score * 0.4) +
        (VitalScore * 0.4) +
        (RespRate_score * 0.10) +
        (consistency_score * 0.10)
    )

    return sleep_score.clip(0, 100)

# Data Preprocessing
def preprocess_data(input):
    from sklearn.impute import KNNImputer
    features = ['total','AwakeDuration', 'CoreDuration', 'REMDuration', 'DeepFraction','REMFraction', 'TimeToBed', 'TimeOutBed',
                'TimeToBedDif',
                'TimeOutBedDif', 'TotalDif', 'averageTimeToBed', 'averageTimeOutBed', 'average_sleep_rhr', 'wristtemp',
                'RespRate', 'HRV', 'avg_wristtemp_60d', 'avg_RespRate_60d', 'avg_average_sleep_rhr_7d', 'avg_HRV_7d',
                'wristtemp_relative_index','RespRate_relative_index','average_sleep_rhr_relative_index','HRV_relative_index','SleepGoal']
    if 'IsNightSleep' in input.columns:
        input = input.loc[input['IsNightSleep']]  # Filter sleep nights
    retained_indices = input.index.copy()
    print(input.columns)
    X = input[features].copy()  # Avoid SettingWithCopyWarning

    # Create mask of missing values
    nan_mask = X.isna()

    # Fit imputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

    # Only update missing values
    X[nan_mask] = X_imputed[nan_mask]
    input.loc[:, features] = X  # safely assign imputed values back

    # Handle missing values with a fill (e.g., fill NaN with 0)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_imputed)
    y = SleepScoreHeuristic(input)
    print(y.describe())
    y.to_csv('score.csv',index=False)

    y_scaled = y_scaler.fit_transform(y.to_frame())

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    return X_tensor, y_tensor, x_scaler, y_scaler , retained_indices, input






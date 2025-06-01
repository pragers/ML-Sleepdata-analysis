import pandas as pd
import numpy as np
from datetime import timedelta
from preprocessing.sleep_analysis_conversion import GetSleepDataFrame


# ------------------------- Utility Functions -------------------------

def removeOutliers(df, columnName, threshold):
    """Remove outliers more efficiently using vectorized operations"""
    if columnName not in df.columns or df[columnName].isna().all():
        return df

    mean = df[columnName].mean(skipna=True)
    std = df[columnName].std(skipna=True)
    outliers = (df[columnName] > mean + threshold * std) | \
               (df[columnName] < mean - threshold * std)
    df.loc[outliers, columnName] = np.nan
    return df


def matchingTimes(df, start_date, end_date):
    """Find matching times using vectorized comparison"""
    return df[(df['startDate'] >= start_date) & (df['startDate'] <= end_date)]


def addData(row, data, columnName):
    """Add data to a row"""
    if not data.empty:
        value_mean = data['value'].mean()
        row[columnName] = value_mean
    else:
        row[columnName] = np.nan
    return row


def weightedMean(measurements):
    """Calculate weighted mean using numpy for speed"""
    measurements = measurements.dropna()
    if len(measurements) == 0:
        return np.nan

    errors = 100 - measurements
    weights = 100 + errors
    return np.average(measurements, weights=weights)


def add_rolling_average(df, date_column, value_columns, window_size=30):
    """Add rolling average with optimized processing"""
    if not value_columns:
        return df

    # Convert once
    df[date_column] = pd.to_datetime(df[date_column])
    df_indexed = df.set_index(date_column)

    # Process all columns at once
    for value_column in value_columns:
        if value_column in df.columns:
            df_indexed[f'avg_{value_column}_{window_size}d'] = df_indexed[value_column].rolling(
                window=f'{window_size}D', min_periods=1).mean()
            df_indexed[f'{value_column}_relative_index'] = (df_indexed[f'{value_column}'] / df_indexed[f'avg_{value_column}_{window_size}d'])
            print(f'Added {value_column}_relative_index to dataframe')

    return df_indexed.reset_index()


# ------------------------- Main Processing -------------------------

def process_sleep_row(row, df_dict):
    """Process each sleep row with optimized access to dataframes"""
    row['startDate'] = pd.to_datetime(row['startDate']).replace(tzinfo=None)
    row['endDate'] = pd.to_datetime(row['endDate']).replace(tzinfo=None)

    sleep_start = row['startDate'] - timedelta(hours=2)
    sleep_end = row['endDate'] + timedelta(hours=2)

    # Process each data source if available
    if 'Rhr' in df_dict and df_dict['Rhr'] is not None:
        matching_rhr = matchingTimes(df_dict['Rhr'], row['startDate'], row['endDate'])
        if not matching_rhr.empty:
            row = addData(row, matching_rhr, 'average_sleep_rhr')

    if 'WristTemp' in df_dict and df_dict['WristTemp'] is not None:
        matching_wristtemp = df_dict['WristTemp'][
            (df_dict['WristTemp']['startDate'] >= sleep_start) &
            (df_dict['WristTemp']['endDate'] <= sleep_end)
            ]
        if not matching_wristtemp.empty:
            row = addData(row, matching_wristtemp, 'wristtemp')

    if 'RespRate' in df_dict and df_dict['RespRate'] is not None:
        matching_RespRate = matchingTimes(df_dict['RespRate'], row['startDate'], row['endDate'])
        if not matching_RespRate.empty:
            row = addData(row, matching_RespRate, 'RespRate')

    if 'HRV' in df_dict and df_dict['HRV'] is not None:
        matching_HRV = matchingTimes(df_dict['HRV'], row['startDate'], row['endDate'])
        if not matching_HRV.empty:
            row = addData(row, matching_HRV, 'HRV')

        # Next day HRV - use pre-computed end date
        end_date = row['endDate'].date()
        matching_HRV1 = df_dict['HRV'][df_dict['HRV']['startDate'].dt.date == end_date]
        if not matching_HRV1.empty:
            row = addData(row, matching_HRV1, 'NextDayHRV')

    if 'HeartRate' in df_dict and df_dict['HeartRate'] is not None:
        # Reuse end_date calculation
        end_date = row['endDate'].date() if 'end_date' not in locals() else end_date
        matching_HeartRate = df_dict['HeartRate'][df_dict['HeartRate']['startDate'].dt.date == end_date]
        if not matching_HeartRate.empty:
            row = addData(row, matching_HeartRate, 'NextDayHeartRate')

    return row


# ------------------------- Main Pipeline -------------------------

def Sleepdata(Sleepdf, Rhrdf=None, WristTempdf=None, RespRatedf=None, HRVdf=None, HeartRatedf=None):
    """Main pipeline function optimized for speed"""
    # Pre-process and sort dataframes only once
    dataframes = {}

    # Convert numeric values once for each dataframe
    for name, df in [
        ('Rhr', Rhrdf),
        ('WristTemp', WristTempdf),
        ('RespRate', RespRatedf),
        ('HRV', HRVdf),
        ('HeartRate', HeartRatedf)
    ]:
        if df is not None:
            df = df.copy()  # Avoid modifying original
            df['startDate'] = pd.to_datetime(df['startDate']).dt.tz_localize(None)
            if 'endDate' in df.columns:
                df['endDate'] = pd.to_datetime(df['endDate']).dt.tz_localize(None)
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            dataframes[name] = df.sort_values(by='startDate')

    # Process sleep dataframe
    SleepWithRhr = Sleepdf.copy()

    # Apply process_sleep_row to each row with dataframes dictionary
    results = []
    for _, row in SleepWithRhr.iterrows():
        results.append(process_sleep_row(row.copy(), dataframes))

    SleepWithRhr = pd.DataFrame(results)

    # Remove outliers where needed
    if 'average_sleep_rhr' in SleepWithRhr.columns:
        SleepWithRhr = removeOutliers(SleepWithRhr, 'average_sleep_rhr', 3)
    if 'HRV' in SleepWithRhr.columns:
        SleepWithRhr = removeOutliers(SleepWithRhr, 'HRV', 4)

    # Add rolling averages in optimized batches
    # First batch
    value_columns = [col for col in ['wristtemp', 'RespRate'] if col in SleepWithRhr.columns]
    if value_columns:
        SleepWithRhr = add_rolling_average(SleepWithRhr, 'startDate', value_columns, window_size=60)

    # Second batch
    value_columns = [col for col in ['average_sleep_rhr', 'HRV'] if col in SleepWithRhr.columns]
    if value_columns:
        SleepWithRhr = add_rolling_average(SleepWithRhr, 'startDate', value_columns, window_size=7)

    # Rename standard columns
    column_mapping = {
        'HKCategoryValueSleepAnalysisAwake': 'AwakeDuration',
        'HKCategoryValueSleepAnalysisAsleepCore': 'CoreDuration',
        'HKCategoryValueSleepAnalysisAsleepDeep': 'DeepDuration',
        'HKCategoryValueSleepAnalysisAsleepREM': 'REMDuration'
    }

    # Only rename columns that exist
    columns_to_rename = {k: v for k, v in column_mapping.items() if k in SleepWithRhr.columns}
    if columns_to_rename:
        SleepWithRhr.rename(columns=columns_to_rename, inplace=True)

    return SleepWithRhr

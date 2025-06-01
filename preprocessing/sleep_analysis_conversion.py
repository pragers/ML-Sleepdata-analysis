import pandas as pd
import numpy as np
from datetime import timedelta, datetime


@pd.api.extensions.register_series_accessor("times")
class TimeMethods:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def seconds_since_midnight(self):
        return (self._obj.dt.hour * 3600 + self._obj.dt.minute * 60 + self._obj.dt.second)


def removeOutliers(df, columnName, threshold):
    mean = df[columnName].mean(skipna=True)
    std = df[columnName].std(skipna=True)
    outliers = (df[columnName] > mean + threshold * std) | (df[columnName] < mean - threshold * std)
    df.loc[outliers, columnName] = np.nan
    return df


def circularMean(values, max=86400):
    radians = values / max * 2 * np.pi
    sin_roll = np.sin(radians).mean()
    cos_roll = np.cos(radians).mean()
    mean_angle = np.mod(np.arctan2(sin_roll, cos_roll), 2 * np.pi)
    return mean_angle / (2 * np.pi) * max


def circular_min_diff(a, b, max_seconds=86400):
    diff = abs(a - b) % max_seconds
    return np.minimum(diff, max_seconds - diff)


def defineNaps(df):
    df['IsNightSleep'] = df['total'].apply(lambda x: x > 4)

    prev_session_end = None
    for i, row in df.iterrows():
        if prev_session_end is not None and prev_session_end.date() == row['endDate'].date():
            df.loc[i, 'IsNightSleep'] = False
        prev_session_end = row['endDate']

    return df


def BedTimeCalc(df):
    df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
    df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')
    df.dropna(subset=['startDate', 'endDate'], inplace=True)

    df['TimeToBed'] = df['startDate'].times.seconds_since_midnight()
    df['TimeOutBed'] = df['endDate'].times.seconds_since_midnight()

    df = df.sort_values('startDate').set_index('startDate')

    df['averageTimeToBed'] = df['TimeToBed'].where(df['IsNightSleep']).rolling(window='14D', min_periods=7).apply(
        circularMean, raw=False)
    df['TimeToBedDif'] = circular_min_diff(df['TimeToBed'], df['averageTimeToBed'])

    df['averageTimeOutBed'] = df['TimeOutBed'].where(df['IsNightSleep']).rolling(window='14D', min_periods=7).apply(
        circularMean, raw=False)
    df['TimeOutBedDif'] = circular_min_diff(df['TimeOutBed'], df['averageTimeOutBed'])

    df['TotalDif'] = df['TimeOutBedDif'] + df['TimeToBedDif']

    return df.reset_index()


def GetSleepDataFrame(df):
    pd.DataFrame.BedTimeCalc = BedTimeCalc
    pd.DataFrame.removeOutliers = removeOutliers
    pd.DataFrame.defineNaps = defineNaps

    df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
    df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')
    df.dropna(subset=['startDate', 'endDate'], inplace=True)

    if df.empty or 'value' not in df.columns:
        return pd.DataFrame()

    df_sorted = df.sort_values(by='startDate')
    sessions_data = []
    threshold = timedelta(minutes=60)

    state_durations = {
        'startDate': None,
        'endDate': None,
        "HKCategoryValueSleepAnalysisInBed": 0,
        "HKCategoryValueSleepAnalysisAsleepUnspecified": 0,
        "HKCategoryValueSleepAnalysisAwake": 0,
        "HKCategoryValueSleepAnalysisAsleepCore": 0,
        "HKCategoryValueSleepAnalysisAsleepDeep": 0,
        "HKCategoryValueSleepAnalysisAsleepREM": 0
    }

    current_session_start = None
    current_session_end = None
    currentsession = -1  # Will be incremented on first entry

    for i, row in df_sorted.iterrows():
        row_value = row['value']
        if pd.isnull(row_value):
            continue

        if current_session_end is None or (row['startDate'] - current_session_end > threshold):
            # Save endDate for the previous session if one exists
            if currentsession >= 0:
                sessions_data[currentsession]['endDate'] = current_session_end

            # Start a new session
            currentsession += 1
            state_durations = {key: 0 for key in state_durations}
            current_session_start = row['startDate']
            current_session_end = row['endDate']
            state_durations['startDate'] = current_session_start
            state_durations[row_value] = (current_session_end - current_session_start).total_seconds() / 3600.0
            state_durations['total'] = (current_session_end - current_session_start).total_seconds() / 3600.0
            if row_value == "HKCategoryValueSleepAnalysisAwake":
                state_durations['WakeEvents'] = 1
            sessions_data.append(state_durations.copy())
        else:
            # Continue current session
            duration = (row['endDate'] - row['startDate']).total_seconds() / 3600.0
            current_session_end = row['endDate']
            sessions_data[currentsession][row_value] += duration
            sessions_data[currentsession]['total'] += duration
            if row_value == "HKCategoryValueSleepAnalysisAwake":
                sessions_data[currentsession]['WakeEvents'] = sessions_data[currentsession].get('WakeEvents', 0) + 1
            sessions_data[currentsession]['endDate'] = current_session_end

        if i == df_sorted.shape[0] - 2:
            sessions_data[currentsession]['endDate'] = current_session_end

    if not sessions_data:
        return pd.DataFrame()

    sessionspd = pd.DataFrame(sessions_data)
    sessionspd['session_id'] = range(1, len(sessionspd) + 1)
    columns = ['session_id'] + [col for col in sessionspd.columns if col != 'session_id']
    sessionspd = sessionspd[columns]

    numeric_columns = sessionspd.select_dtypes(include=['float64']).columns
    if 'HKCategoryValueSleepAnalysisInBed' in numeric_columns:
        numeric_columns = numeric_columns.drop('HKCategoryValueSleepAnalysisInBed', errors='ignore')


    sessionspd['DeepFraction'] = sessionspd['HKCategoryValueSleepAnalysisAsleepDeep'] / sessionspd['total']
    sessionspd['REMFraction'] = sessionspd['HKCategoryValueSleepAnalysisAsleepREM'] / sessionspd['total']

    sessionspd = sessionspd.defineNaps()
    sessionspd = sessionspd.BedTimeCalc()

    return sessionspd
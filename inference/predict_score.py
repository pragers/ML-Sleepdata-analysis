import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models.vital_prediction_model import SleepQualityRegressor
from preprocessing.xml_to_dataframe_converter import xml_to_dataframe
from preprocessing.data_preprocessing import preprocess_data, SleepScoreHeuristic
from preprocessing.data_type_sorter import split_csv_by_type_fast
from datetime import timedelta, datetime
from preprocessing.process_sleep_data import Sleepdata
from preprocessing.sleep_analysis_conversion import GetSleepDataFrame

def ProcessData(data):
    Features = ['total','DeepFraction','TotalDif','wristtemp','RespRate','HRV','NextDayHRVPred', 'NextDayHeartRatePred']
    data_filtered = data.dropna(how='all', subset=[f for f in Features if f != 'total'])
    print(data_filtered)
    X= data_filtered[Features]
    from sklearn.impute import KNNImputer
    imputer_x = KNNImputer(n_neighbors=5)
    X_imputed = imputer_x.fit_transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_scaled_df = pd.DataFrame(X_scaled, columns=Features)
    return X_tensor,data_filtered

def GetScore(inputfile):
    df = xml_to_dataframe([inputfile])
    data1 = pd.read_csv('TrainingData.csv')
    df['startDate'] = pd.to_datetime(df['startDate']).dt.tz_localize(tz=None)
    df['endDate'] = pd.to_datetime(df['endDate']).dt.tz_localize(tz=None)
    # latest_date = datetime.now()
    # cutoff_date = latest_date - timedelta(days=30)
    # recent_data = df[df['startDate'] >= cutoff_date]

    data = split_csv_by_type_fast(df)

    if data.get('SleepAnalysis') is not None:
        data = Sleepdata(GetSleepDataFrame(data['SleepAnalysis']), data.get('HeartRate'),
                         data.get('AppleSleepingWristTemperature'), data.get('RespiratoryRate'),
                         data.get('HeartRateVariabilitySDNN'), data.get('RestingHeartRate'))
        #model = SleepQualityRegressor()
        data['SleepGoal'] = 8.0
        data = pd.concat([data, data1], ignore_index=True)
        X_tensor, y_tensor, x_scaler, y_scaler, retained_indices, imputed_data = preprocess_data(data)
        # with torch.no_grad():
        #     outputs = model(X_tensor)
        # predictions = y_scaler.inverse_transform(outputs.numpy())
        # data.loc[retained_indices, ["NextDayHRVPred", "NextDayHeartRatePred"]] = predictions
        # model2 =  torch.load('models/trained_models/full_model.pth',weights_only=False)
        # model2.eval()
        # x,data_filtered = ProcessData(data)
        # with torch.no_grad():
        #     outputs = model2(x)
        # predicted_scores = outputs.detach().numpy().flatten()
        # predicted_scores = np.clip(predicted_scores, 0, 100)
        # data_with_scores = data.copy()
        # data_with_scores.loc[data_filtered.index, 'SleepScore'] = predicted_scores
        # data_with_scores.to_csv('Sleep/SleepDataWithScores1.csv', index=False)


        #return last_day_data
        imputed_data = imputed_data.copy()
        imputed_data['SleepScore'] = SleepScoreHeuristic(imputed_data)
        imputed_data.to_csv('TrainingData.csv',index=False)
        last_date = imputed_data['startDate'].max()
        last_day_data = imputed_data[imputed_data['startDate'] == last_date]
        return last_day_data




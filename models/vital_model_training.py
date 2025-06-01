from datetime import datetime

import numpy as np

from preprocessing.data_preprocessing import *
from preprocessing.process_sleep_data import Sleepdata
from preprocessing.sleep_analysis_conversion import GetSleepDataFrame
from preprocessing.xml_to_dataframe_converter import xml_to_dataframe
from preprocessing.data_type_sorter import split_csv_by_type_fast
from vital_prediction_model import SleepQualityRegressor, train_model, make_predictions


##
def GetSleepData(inputfile):

    df = split_csv_by_type_fast(xml_to_dataframe([inputfile]))
    if df.get('SleepAnalysis') is not None:
        data = Sleepdata(GetSleepDataFrame(df['SleepAnalysis']),df.get('HeartRate'),df.get('AppleSleepingWristTemperature'),df.get('RespiratoryRate'),df.get('HeartRateVariabilitySDNN'),df.get('RestingHeartRate'))
        return data
    else:
        return df

# Example: Using DataFrame `df`
def TrainModel(data,Epochs):

    X_tensor, y_tensor, x_scaler, y_scaler, retained_indices, imputed_data = preprocess_data(data)


    # Create the model
    model = SleepQualityRegressor(input_dim=X_tensor.shape[1])

    # Train the model
    model = train_model(X_tensor, y_tensor, model, num_epochs=Epochs,updateFreq = 100)

    # Make predictions
    predictions = make_predictions(model, X_tensor, y_scaler)
    np.set_printoptions(suppress=True)
    print("Predicted Values (Inverse Scaled):", predictions)

    # Inverse-transform the original targets for a fair comparison
    y_true = y_scaler.inverse_transform(y_tensor.numpy())
    y_pred = predictions  # Already inverse-transformed in your pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    # Calculate errors
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")

    df_predictions = pd.DataFrame(predictions, columns=["NextDayHRV", "NextDayHeartRate"])
    data.loc[retained_indices, imputed_data.columns] = imputed_data
    data.loc[retained_indices, ["NextDayHRVPred", "NextDayHeartRatePred"]] = predictions
    torch.save(model, 'full_model1.pth')
    data.to_csv('DataTestWPred.csv')

# data1 = pd.read_csv('../outputs/Test.csv')
# data2 = pd.read_csv('../outputs/Test1.csv')
# TotalData = pd.concat([data1,data2],ignore_index=True)
# TrainModel(TotalData,Epochs=2000)

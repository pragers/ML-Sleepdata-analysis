import pandas as pd

def split_csv_by_type(df):
    # Initialize a dictionary to store DataFrames by type
    dataframes = {}

    for _, row in df.iterrows():
        record_type = row.get('type')
        if not record_type:
            continue  # Skip rows without a 'type'

        # Remove the prefixes 'HKQuantityTypeIdentifier' and 'HKDataType'
        record_type = record_type.replace('HKQuantityTypeIdentifier', '').replace('HKDataType', '')

        # Clean the record_type to make it safe for column names
        safe_record_type = record_type.replace(":", "_").replace("/", "_")

        if safe_record_type not in dataframes:
            # Initialize a new DataFrame for this record type
            dataframes[safe_record_type] = pd.DataFrame(columns=df.columns)

        # Convert row to DataFrame and drop empty or all-NA columns
        row_df = pd.DataFrame([row]).dropna(axis=1, how='all')  # Drop columns where all values are NaN

        # Drop columns where all values are NaN (to handle future behavior change)
        row_df = row_df.dropna(axis=1, how='all')

        # Use pd.concat to append the row to the corresponding DataFrame
        dataframes[safe_record_type] = pd.concat([dataframes[safe_record_type], row_df], ignore_index=True, verify_integrity=True)

    return dataframes
def split_csv_by_type_fast(df, output_dir="./split_data"):
    # Clean up type names
    df.loc[:, 'type'] = df['type'].str.replace('HKQuantityTypeIdentifier', '', regex=False)
    df.loc[:, 'type'] = df['type'].str.replace('HKDataType', '', regex=False)
    df.loc[:, 'type'] = df['type'].str.replace('HKCategoryTypeIdentifier', '', regex=False)
    df.loc[:, 'type'] = df['type'].str.replace(r'[:/]', '_', regex=True)
    df.loc[:, 'startDate'] = pd.to_datetime(df['startDate']).dt.tz_localize(tz=None)
    df.loc[:, 'endDate'] = pd.to_datetime(df['endDate']).dt.tz_localize(tz=None)


    # Split and save
    grouped = dict(tuple(df.groupby('type')))
    #for typename, group in grouped.items():
    #    filename = f"{output_dir}/{typename}.csv"
    #    group.to_csv(filename, index=False)

    return grouped  # Optional: return in case you want to use it further
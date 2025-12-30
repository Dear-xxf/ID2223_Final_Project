import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import hopsworks
import joblib
import os


def process_dhs_data(file_dhs_path: str, indicators: List[str]) -> pd.DataFrame:
    """
    Load DHS daily report data, select key columns, and perform date processing.
    Args:
        file_dhs_path: Path to the DHS CSV file.
    Returns:
        pd.DataFrame: The pre-processed DHS data.
    """
    try:
        df_dhs = pd.read_csv(file_dhs_path)
    except Exception as e:
        raise IOError(f"DHS file failed to load: {e}")

    df_dhs = df_dhs[indicators].copy()

    df_dhs['Date of Census'] = pd.to_datetime(df_dhs['Date of Census'])
    df_dhs['join_year'] = df_dhs['Date of Census'].dt.year

    df_dhs = df_dhs.sort_values(by='Date of Census').reset_index(drop=True)
    df_dhs.columns = [col.lower().replace(' ', '_') for col in df_dhs.columns]

    return df_dhs

def clean_numeric_value(value: Union[str, float]) -> float:
    if isinstance(value, str):
        return float(value.replace('"', '').replace(',', ''))
    return float(value)


def process_economic_data(file_econ_path: str, indicators: List[str]) -> pd.DataFrame:
    """
    Load economic forecast data, filter indicators, average the forecast values for the same year, convert to a wide format, and clean up column names.
    Args:
        file_econ_path: Path to the economic forecast CSV file.
    indicators: List of economic indicators to extract.
    Returns:
        pd.DataFrame: Aggregated, wide-formatted, and cleaned economic data.
    """
    try:
        df_econ = pd.read_csv(file_econ_path)
    except Exception as e:
        raise IOError(f"Economic forecast file failed to load: {e}")

    df_econ_filtered = df_econ[
        (df_econ['ECONOMIC INDICATOR'].isin(indicators))
    ].copy()

    df_econ_filtered['FORECAST_VALUE_CLEANED'] = df_econ_filtered['FORECAST VALUE'].apply(clean_numeric_value)

    df_econ_aggregated = df_econ_filtered.groupby(
        ['REFERENCE YEAR', 'ECONOMIC INDICATOR']
    )['FORECAST_VALUE_CLEANED'].mean().reset_index()

    df_econ_pivot = df_econ_aggregated.pivot_table(
        index='REFERENCE YEAR',
        columns='ECONOMIC INDICATOR',
        values='FORECAST_VALUE_CLEANED'
    ).reset_index()

    df_econ_pivot.rename(columns={'REFERENCE YEAR': 'join_year'}, inplace=True)

    new_cols_map = {}
    for col in df_econ_pivot.columns:
        if col != 'join_year':
            sanitized_col = col.lower()
            sanitized_col = sanitized_col.replace(' ', '_').replace('.', '_').replace('-', '_')
            while '__' in sanitized_col:
                sanitized_col = sanitized_col.replace('__', '_')

            new_cols_map[col] = sanitized_col

    df_econ_pivot.rename(columns=new_cols_map, inplace=True)

    df_econ_pivot.to_csv(ECON_PIVOT_PATH, index=False)

    return df_econ_pivot

def create_combined_features(df_dhs: pd.DataFrame, df_econ_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate DHS and economic data, and add time series and lagged features.
    Args:
        df_dhs: Processed DHS DataFrame.
        df_econ_pivot: Processed economic forecast DataFrame (wide format).
    Returns:
        pd.DataFrame: The final feature set df_combined for machine learning.
    """
    # merge data
    df_combined = pd.merge(
        df_dhs,
        df_econ_pivot,
        on='join_year',
        how='left'
    )

    # Remove secondary join column
    df_combined.drop(columns=['join_year'], inplace=True)

    df_combined['year'] = df_combined['date_of_census'].dt.year
    df_combined['month'] = df_combined['date_of_census'].dt.month
    df_combined['day_of_week'] = df_combined['date_of_census'].dt.dayofweek
    df_combined['is_weekend'] = (df_combined['day_of_week'] >= 5).astype(int)

    # Create hysteresis features
    df_combined['children_lag_1'] = df_combined['total_children_in_shelter'].shift(1)
    df_combined['children_lag_7'] = df_combined['total_children_in_shelter'].shift(7)

    ADULT_LAG_FEATURES = [
        'total_adults_in_shelter',
        'single_adult_men_in_shelter',
        'single_adult_women_in_shelter',
        'families_with_children_in_shelter'
    ]

    for col in ADULT_LAG_FEATURES:
        lag_col_name = f"{col}_lag_1"
        df_combined[lag_col_name] = df_combined[col].shift(1)

    df_combined = df_combined.dropna().reset_index(drop=True)

    return df_combined


if __name__ == '__main__':
    DHS_FILE_PATH = "../data/DHS_Daily_Report_20251210.csv"
    ECON_FILE_PATH = "../data/New_York_City_Forecasts_of_Selected_Economic_Indicators_20251212.csv"
    ECON_PIVOT_PATH = "../data/econ_pivot_cache.csv"

    DHS_INDICATORS = [
        'Date of Census',
        'Total Adults in Shelter',
        'Single Adult Men in Shelter',
        'Single Adult Women in Shelter',
        'Families with Children in Shelter',
        'Total Children in Shelter'
    ]

    # Selected economic indicators
    ECNO_INDICATORS = [
        'US Non-Agricultural Employment',
        'NYC Wage Rate',
        'NYC Personal Income',
        'NYC Real Gross City Product',
        'US Consumer Price Index'
    ]

    try:
        # Processing DHS data
        df_dhs = process_dhs_data(DHS_FILE_PATH, DHS_INDICATORS)
        # Processing economic data
        df_econ_pivot = process_economic_data(ECON_FILE_PATH, ECNO_INDICATORS)
        # Connect and create the final feature set
        df_combined_features = create_combined_features(df_dhs, df_econ_pivot)
        print("\nThe final feature set was created successfully. (df_combined_features)。")
        print(f"Total num of rows: {len(df_combined_features)}")
        print("Preview:")
        print(df_combined_features.head())
        print("Structure:")
        df_combined_features.info()

    except Exception as e:
        print(f"\nFatal error: Data processing interrupted.。{e}")


    HOPSWORKS_PROJECT = "ID2223_finn"
    API_KEY = os.environ.get("HOPSWORKS_API_KEY")

    FEATURE_GROUP_NAME = 'dhs_shelter_children_features'
    FEATURE_GROUP_VERSION = 1

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=API_KEY
    )
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Daily shelter data combined with annual NYC averaged economic forecasts for children count prediction.",
        primary_key=['date_of_census'],
        event_time='date_of_census',
    )

    feature_group.insert(
        df_combined_features,
        write_options={"wait_for_job": True}
    )
import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
import joblib
from datetime import datetime, timedelta
import numpy as np
from data_processing import create_combined_features

from sodapy import Socrata
import pytz

HOPSWORKS_PROJECT = "ID2223_finn"
API_KEY = "qIowjnpw6LP09gpU.IE7Mx9TexhndcJdBAN6kU3eYgI56pgOaGtIqCZizr5cUrlgLsFNvPs7xklcKqkSx"

# Hopsworks Feature Group & Model Info
FEATURE_GROUP_NAME = 'dhs_shelter_children_features'
FEATURE_GROUP_VERSION = 1
TARGET_COLUMN = 'total_children_in_shelter'
MODEL_PATH = '../model/xgboost_shelter_model.pkl'
DATASET_ID = "k46n-sa2m"
ECON_PIVOT_PATH = "../data/econ_pivot_cache.csv"

DHS_INDICATORS = [
        'date_of_census',
        'total_adults_in_shelter',
        'single_adult_men_in_shelter',
        'single_adult_women_in_shelter',
        'families_with_children_in_shelter',
        'total_children_in_shelter'
    ]

project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=API_KEY
    )
fs = project.get_feature_store()


def fetch_latest_data(date_str: str) -> pd.DataFrame:
    try:
        client = Socrata("data.cityofnewyork.us", None)
        socrata_results = client.get(
            DATASET_ID,
            where=f"date_of_census = '{date_str}'",
            limit=2
        )

        if not socrata_results:
            print(f"❌ 警告: API 未返回 {date_str} 的数据。")
            return pd.DataFrame()

        return pd.DataFrame.from_records(socrata_results)

    except Exception as e:
        print(f"❌ Socrata API 调用失败: {e}")
        return pd.DataFrame()


def to_utc(dt):
    if pd.isna(dt):
        return dt
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.tz_localize('UTC')
    return dt.tz_convert('UTC')

def run_daily_update_and_prediction():
    today = (datetime.now()- timedelta(days=1)).strftime('%Y-%m-%d')
    today_date = datetime.strptime(today, '%Y-%m-%d') - timedelta(hours=12)
    print(f"当前任务日期 (T日): {today}")

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )


    query_today = feature_group.select_all().filter(
        (feature_group.date_of_census >= today_date)
    )
    if not query_today.read().empty:
        print('Today data already exists. Skipping update.')
        return pd.DataFrame()

    df_new_raw = fetch_latest_data(today)

    if df_new_raw.empty:
        return pd.DataFrame()

    df_new_processed = df_new_raw[DHS_INDICATORS].copy()
    df_new_processed['date_of_census'] = pd.to_datetime(df_new_processed['date_of_census'])
    df_new_processed['join_year'] = df_new_processed['date_of_census'].dt.year
    df_new_processed = df_new_processed.sort_values(by='date_of_census').reset_index(drop=True)

    query_start_date = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=8)).strftime('%Y-%m-%d')

    # 获取历史数据 (T-7 到 T-1)
    query = feature_group.select_all().filter(feature_group.date_of_census >= query_start_date)
    df_history = query.read()

    df_econ_pivot = pd.read_csv(ECON_PIVOT_PATH)
    df_combined_new = pd.merge(
        df_new_processed,
        df_econ_pivot,
        on='join_year',
        how='left'
    )
    df_combined_new.drop(columns=['join_year'], inplace=True)
    df_combined_new['year'] = df_combined_new['date_of_census'].dt.year
    df_combined_new['month'] = df_combined_new['date_of_census'].dt.month
    df_combined_new['day_of_week'] = df_combined_new['date_of_census'].dt.dayofweek
    df_combined_new['is_weekend'] = (df_combined_new['day_of_week'] >= 5).astype(int)

    df_combined_new = pd.concat([df_history, df_combined_new], ignore_index=True)

    df_combined_new['children_lag_1'] = df_combined_new['total_children_in_shelter'].shift(1)
    df_combined_new['children_lag_7'] = df_combined_new['total_children_in_shelter'].shift(7)
    df_combined_new['single_adult_men_in_shelter_lag_1'] = df_combined_new['single_adult_men_in_shelter'].shift(1)
    df_combined_new['single_adult_women_in_shelter_lag_1'] = df_combined_new['single_adult_women_in_shelter'].shift(1)
    df_combined_new['total_adults_in_shelter_lag_1'] = df_combined_new['total_adults_in_shelter'].shift(1)
    df_combined_new['families_with_children_in_shelter_lag_1'] = df_combined_new['families_with_children_in_shelter'].shift(1)
    df_combined_new = df_combined_new.dropna().reset_index(drop=True)

    df_combined_new['date_of_census'] = df_combined_new['date_of_census'].apply(to_utc)

    BIGINT_COLS = [
        'total_adults_in_shelter', 'single_adult_men_in_shelter',
        'single_adult_women_in_shelter', 'families_with_children_in_shelter',
        'total_children_in_shelter', 'total_individuals_in_shelter'
    ]
    DOUBLE_COLS = ['children_lag_1', 'children_lag_7',
                   'total_adults_in_shelter_lag_1', 'single_adult_men_in_shelter_lag_1',
                   'single_adult_women_in_shelter_lag_1', 'families_with_children_in_shelter_lag_1']
    for col in BIGINT_COLS:
        if col in df_combined_new.columns:
            df_combined_new[col] = pd.to_numeric(df_combined_new[col], errors='coerce').astype(pd.Int64Dtype())
    for col in DOUBLE_COLS:
        if col in df_combined_new.columns:
            df_combined_new[col] = df_combined_new[col].astype('float64')


    feature_group.insert(
        df_combined_new
    )
    return df_combined_new


def predict_children_in_shelter(df: pd.DataFrame):
    model = joblib.load(MODEL_PATH)

    EXCLUDE_COLUMNS = ['date_of_census', 'total_children_in_shelter',
                       'total_adults_in_shelter', 'single_adult_men_in_shelter', 'single_adult_women_in_shelter',
                       'families_with_children_in_shelter']

    X_new = df.drop(columns=EXCLUDE_COLUMNS)
    prediction = model.predict(X_new)
    df['predicted_children_in_shelter'] = int(round(prediction[0]))
    print("预测结果:")
    print(df[['date_of_census', 'total_children_in_shelter']])
    print(df['predicted_children_in_shelter'])

    feature_group = fs.get_feature_group(
        name="dhs_shelter_children_predictions",
        version=1
    )

    feature_group.insert(
        df[['date_of_census', 'predicted_children_in_shelter', 'total_children_in_shelter']],
        write_options={"wait_for_job": True}
    )

    return df


if __name__ == '__main__':
    df = run_daily_update_and_prediction()
    if not df.empty:
        df_with_predictions = predict_children_in_shelter(df)

    # data = {
    #     'date_of_census': [pd.to_datetime('2025-12-27 00:00:00+00:00')],
    #     'predicted_children_in_shelter': [30692],
    #     'total_children_in_shelter': [30685]
    # }
    # manual_df = pd.DataFrame(data)
    # feature_group = fs.get_feature_group(
    #     name="dhs_shelter_children_predictions",
    #     version=1
    # )
    # feature_group.insert(
    #     manual_df,
    #     write_options={"wait_for_job": True}
    # )
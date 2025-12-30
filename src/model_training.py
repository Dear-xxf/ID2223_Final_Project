import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

import joblib

HOPSWORKS_PROJECT = "ID2223_finn"
API_KEY = os.environ.get("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = 'dhs_shelter_children_features'
FEATURE_GROUP_VERSION = 1
TARGET_COLUMN = 'total_children_in_shelter'

MODEL_NAME = "dhs_children_shelter_predictor_xgboost"
MODEL_VERSION = 1
MODEL_DIR = '../model/'


def train_and_save_xgboost_model():
    # -----------------------------------------------------
    # Step 1: Connect to HopsWorks and obtain the Feature Store
    # -----------------------------------------------------
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=API_KEY
    )
    fs = project.get_feature_store()

    # -----------------------------------------------------
    # Step 2: Read data from Feature Group
    # -----------------------------------------------------

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df_full = feature_group.read()
    print(f"Successfully read {len(df_full)} rows of data.")

    # -----------------------------------------------------
    # Step 3: prepare features and target variable
    # -----------------------------------------------------

    # Exclude columns not used for model training:
    # - date_of_census: Timestamps are not used as features.
    # - target_column: Target variable needs to be separated

    EXCLUDE_COLUMNS = ['date_of_census', TARGET_COLUMN,
                       'total_adults_in_shelter', 'single_adult_men_in_shelter', 'single_adult_women_in_shelter',
                       'families_with_children_in_shelter']

    X = df_full.drop(columns=EXCLUDE_COLUMNS)
    y = df_full[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False,
        random_state=42
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Train set size: {len(X_test)}")

    # -----------------------------------------------------
    # Step 4: Training the XGBoost regression model
    # -----------------------------------------------------

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Model training is complete.")

    # -----------------------------------------------------
    # Step 5: Model Evaluation
    # -----------------------------------------------------

    y_pred = model.predict(X_test)
    # Calculate MSE, RMSE, and MAE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test set MSE: {mse:.2f}")
    print(f"Test set RMSE: {rmse:.2f}")
    print(f"Test set MAE: {mae:.2f}")

    # -----------------------------------------------------
    # Step 6: Analyze the importance of features
    # -----------------------------------------------------
    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    print(feature_importances.head(5))

    register_model_to_hopsworks(model=model, X_train=X_train, rmse=rmse, mae=mae)



def register_model_to_hopsworks(model, X_train, rmse, mae):
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=API_KEY
    )
    joblib.dump(model, MODEL_DIR + 'xgboost_shelter_model.pkl')

    model_registry = project.get_model_registry()
    registered_model = model_registry.python.create_model(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        description="XGBoost model predicting total children in shelter based on DHS daily data, economic forecasts, and lag features.",
        input_example=X_train.head(1),
        metrics={
            "rmse": float(rmse),
            "mae": float(mae)
        },
    )
    registered_model.save(MODEL_DIR)


if __name__ == '__main__':
    train_and_save_xgboost_model()

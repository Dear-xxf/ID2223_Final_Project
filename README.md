# NYC DHS Shelter Children Population Prediction

New York City Shelter Child Population Real-Time Prediction System

## Objectives

In this project, we built a serverless machine learning system to predict the total number of sheltered children within the New York City Department of Homeland Security (DHS) containment system.

We integrated two key datasets from the NYC Open Data platform(https://opendata.cityofnewyork.us/):

[DHS Daily Report]: Contains daily containment census data from 2013 to the present, updated daily.(https://data.cityofnewyork.us/Social-Services/DHS-Daily-Report/k46n-sa2m/about_data)

[NYC Economic Indicators]: Contains official forecasts for key economic indicators such as employment, personal income, and CPI.(https://data.cityofnewyork.us/City-Government/New-York-City-Forecasts-of-Selected-Economic-Indic/xatq-cxeq/about_data)

The data was automatically acquired via the Socrata API and integrated using the sodapy library.

The system employs a serverless architecture, utilizing GitHub Actions for automated scheduling, Hopsworks as the feature store, and Hugging Face as the front-end display.

## Methodology

### 1. Data Preprocessing

In `preprocessor.py`, we performed deep cleaning and feature transformation on the raw data to construct input variables with high predictive power:

- **Handling Outliers and Missing Values:** Due to the extremely high data quality of DHS daily reports, the proportion of missing values is very low. To ensure the absolute continuity of the time series, we directly remove rows containing null values, ensuring that the model learns a true and consistent trend.
- **Multidimensional Demographic Feature Extraction**: In addition to date information, we extracted the following key demographic parameters as the core support for model prediction:
  - `Total Adults in Shelter` 
  - `Single Adult Men/Women in Shelter` 
  - `Families with Children in Shelter` 
  - `Total Children in Shelter` (**Target variable**)
- **Cross-Frequency Economic Feature Fusion:** New York City economic forecast data is typically released quarterly. We transformed this into daily features, achieving a precise alignment between macroeconomic indicators and microeconomic data.
- **Lag Features**: Introduced child number data for **t-1** and **t-7** (yesterday and last week's same period).

### 2. Feature Pipeline

We have built a fully automated feature engineering pipeline to ensure data real-time performance and consistency:

- **Automatic Scheduling:** Utilizing **GitHub Actions**, the system is scheduled to run automatically at **23:00 CET** daily. This time point is chosen after the data source is updated, ensuring the model always processes the latest social data.

- **Storage and Synchronization:** The script automatically retrieves raw data from the API, preprocesses it, and then synchronizes it to the **Hopsworks Feature Store**. Currently, the feature table has accumulated thousands of historical entries, providing ample context for online inference.

### 3. Training Pipeline

Model training was performed using `training_pipeline.py`, aiming to capture complex nonlinear socioeconomic relationships:

- **Algorithm Selection:** XGBoost Regressor was used.

- **Training Configuration:** The dataset was randomly split into **80% training set** and **20% test set**.

- **Model Performance (Evaluation Metrics):**

The model's evaluation metrics on the independent test set are as follows:

- **MSE (Mean Squared Error):** 5852.19

- **RMSE (Root Mean Squared Error):** 76.50

- **MAE (Mean Absolute Error):** 50.54

- **Top 5 Feature Importance:**

Analysis shows that historical data (Lag) plays a dominant role in the prediction results, while economic indicators also contribute significantly to predictive power:

| **Feature Name**                 | **F-Score** |
| -------------------------------- | ----------- |
| **children_lag_1**               | **0.5847**  |
| **children_lag_7**               | **0.3788**  |
| **families_with_children_lag_1** | **0.0197**  |
| **nyc_personal_income**          | **0.0082**  |
| **single_adult_women_lag_1**     | **0.0060**  |

- **Model Storage and Version Control:** Trained models are saved in `.pkl` format and registered in the **Hopsworks Model Registry**. This allows for easy model version rollback and canary releases, ensuring stability at the inference end.

## Deliverables

This project ultimately delivered an interactive monitoring platform deployed on **Hugging Face Spaces**. This platform transforms complex model inference results into an intuitive visualization interface, allowing policymakers and social observers to monitor data trends in real time.

### 1. Real-time Prediction Display

The top of the user interface prominently displays the predicted number of children in care **for the current date**:

- **Dynamic Values**: The system automatically retrieves data stored in Hopsworks, calculates and displays the estimated total number of children for today in real time.

- **Data Status**: The last update time of the data is clearly indicated to ensure the timeliness of the information.

- **Technical Implementation**: Gradio's HTML components are used for styling and aesthetic enhancement, making the prediction results readily understandable.

### 2. Historical Trend Chart

To verify the model's reliability and demonstrate the changing patterns of inhabitant numbers, a dynamic line chart is integrated into the UI:

- **Comparison Dimensions**: The chart simultaneously plots historical curves for both **"Actual" and **"Predicted" values**.

- **Features**:
  - **Time Series Tracking**: Users can review the model's fit over a past period.

  - **Trend Recognition**: Interactive charts implemented using Plotly allow users to zoom or hover to view detailed data points for specific dates.

  - **Pattern Discovery**: The chart clearly shows the changing trends in inhabitant numbers influenced by seasonal fluctuations and economic characteristics.


### 3. Access Address

You can directly access and experience this project through the following link: https://huggingface.co/spaces/Dear-xxf/DHS_prediction
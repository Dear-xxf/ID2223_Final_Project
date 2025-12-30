import gradio as gr
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv("../.env")

HOPSWORKS_PROJECT = "ID2223_finn"
API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=API_KEY
    )
fs = project.get_feature_store()

# ----------------------------------------------------------------------
# 1. Configuration & Mock Data
# ----------------------------------------------------------------------

# HTML template
PREDICTION_HTML_TEMPLATE = """
<div style="text-align: center; padding: 25px; border: 3px solid #28a745; border-radius: 12px; background-color: #f7fff7; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h2 style="color: #007bff; margin-bottom: 10px;">🏢 Predicted number of children in New York shelters</h2>
    <p style="font-size: 1.2em; color: #555;">The model estimates the number of children admitted on the predicted date:</p>

    <div style="margin: 20px 0;">
        <span style="font-size: 4em; font-weight: bold; color: #28a745;">{prediction_count}</span> 
        <span style="font-size: 2em; color: #333;">人</span>
    </div>

    <p style="font-size: 1.1em; color: #333;">Predict target date:
        <strong style="color: #000; font-size: 1.4em; font-weight: 700;">{target_date}</strong>
    </p>

    <p style="font-size: 0.8em; color: #999; border-top: 1px dashed #ddd; padding-top: 10px;">
        Data display updated on: {last_update_time}
    </p>
</div>
"""

def get_real_historical_data():
    # 1. get Feature Group
    fg = fs.get_feature_group(
        name="dhs_shelter_children_predictions",
        version=1
    )

    df = fg.read()

    if df.empty:
        print("Warning: The Feature Group is empty and there is no data to read!")
        return pd.DataFrame()

    column_mapping = {
        'date_of_census': 'date',
        'predicted_children_in_shelter': 'Predicted',
        'total_children_in_shelter': 'Actual'
    }
    df = df.rename(columns=column_mapping)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    print(df["Predicted"].head())
    return df


# ----------------------------------------------------------------------
# 2. Data Processing Functions
# ----------------------------------------------------------------------

def load_prediction_data():
    df = get_real_historical_data()
    if df.empty:
        return {
            "date": "未获取到数据",
            "predicted_children": 0,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    latest_row = df.iloc[-1]
    real_today_data = {
        "date": latest_row['date'].strftime('%Y-%m-%d'),
        "predicted_children": int(latest_row['Predicted']),
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return real_today_data


def display_prediction_html(data):
    if not data:
        return "Unable to load forecast data."

    output_html = PREDICTION_HTML_TEMPLATE.format(
        prediction_count=data['predicted_children'],
        target_date=data['date'],
        last_update_time=data['last_updated']
    )

    return output_html


def plot_comparison_chart():
    try:
        df = get_real_historical_data()
    except Exception as e:
        print(f"Failed to load historical data: {e}")
        return gr.Plot.update(figure=None)

    # Convert data from wide format to long format
    df_long = pd.melt(df, id_vars=['date'], value_vars=['Predicted', 'Actual'],
                      var_name='Type', value_name='Children Count')

    # Drawing time series charts
    fig = px.line(
        df_long,
        x='date',
        y='Children Count',
        color='Type',
        title='Recent forecasts vs. actual number of people',
        labels={'Children Count': 'Children Count', 'date': 'date', 'Type': 'Type'},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title="date",
        yaxis_title="Children Count",
        hovermode="x unified"
    )

    return fig


# ----------------------------------------------------------------------
# 3. Gradio Blocks Interface Build (Gradio Blocks Interface)
# ----------------------------------------------------------------------

with gr.Blocks(title="Daily forecasts displayed automatically") as demo:
    gr.Markdown("## New York DHS Shelter Child Population Trend Forecast")

    with gr.Row():
        prediction_display = gr.HTML(label="Latest forecast results")

    gr.Markdown("---")
    gr.Markdown("### Recent Predictive Performance Analysis")

    with gr.Row():
        comparison_chart = gr.Plot(label="Comparison of predicted vs. actual number of people")

    demo.load(
        fn=load_prediction_data,
        inputs=None,
        outputs=prediction_display,
        queue=False
    ).then(
        fn=display_prediction_html,
        inputs=prediction_display,
        outputs=prediction_display
    )

    demo.load(
        fn=plot_comparison_chart,
        inputs=None,
        outputs=comparison_chart,
        queue=False
    )

if __name__ == "__main__":
    print("The Gradio application is starting up...")
    demo.launch()
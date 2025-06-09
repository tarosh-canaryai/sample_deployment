import streamlit as st
import json
import urllib.request
import urllib.error
import pandas as pd
from datetime import date, timedelta
from io import BytesIO
import plotly.express as px
import numpy as np
import hashlib

# API endpoint and key (key fetched securely from Streamlit secrets)
API_URL = "https://attrition-pred-v1-debug-score.eastus2.inference.ml.azure.com/score"
API_KEY = st.secrets["API_KEY"] 

# Required columns for making predictions
REQUIRED_API_COLUMNS = [
    "Gender",
    "Marital status",
    "Hourly comp",
    "Race",
    "State",
    "Age",
    "Hire Date",
    "Employee type",
    "Rehire",
    "Home Department"
]

# Columns to show in final output
DISPLAY_COLUMNS = REQUIRED_API_COLUMNS + [
    "Predicted Status",
    "Probability (Active)",
    "Probability (Terminated)",
    "Attrition Risk Level",
]

st.set_page_config(layout="wide")
st.title("Employee Attrition Predictor ")

# UI layout: Two columns
col_input, col_output = st.columns(2)

with col_input:
    st.header("Upload Employee Data (Excel)")
    st.write("Or use the manual form below for single predictions.")

    # Excel file upload for batch predictions
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help=f"Upload an Excel file containing employee data for batch prediction. "
             f"Ensure your Excel sheet has columns named: {', '.join(REQUIRED_API_COLUMNS)}."
    )

    st.markdown("---")
    st.header("Or Enter Single Employee Details")
    with st.form("attrition_form_single"):
        # Manual input fields for single employee prediction
        gender = st.selectbox("Gender", ["M", "F", "Other"], key="single_gender")
        marital_status = st.selectbox("Marital Status", ["S", "M"], key="single_marital")
        hourly_comp = st.number_input("Hourly Compensation", min_value=0.0, step=0.5, value=5.0, key="single_hourly")
        race = st.selectbox("Race", ["B","H","T","W","X"], key="single_race")
        state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                                    "DC" ], key="single_state")
        age = st.number_input("Age", min_value=16, max_value=100, step=1, value=20, key="single_age")
        
        # Default hire date set to ~6 months ago
        six_months_ago = date.today() - timedelta(days=6*30) 
        hire_date = st.date_input("Hire Date", value=six_months_ago, key="single_hire_date")

        employee_type = st.selectbox("Employee Type", ["Full time", "Part time", "Temporary"], key="single_emp_type")
        rehire = st.selectbox("Rehire", ["Yes", "No"], key="single_rehire")
        home_dept = st.text_input("Home Department", key="single_home_dept")

        # Submit button for single prediction
        submit_single = st.form_submit_button("Predict Single Employee")

with col_output:
    st.header("Prediction Results")
    results_placeholder = st.empty() 

    if not uploaded_file and not submit_single:
        results_placeholder.info("Upload an Excel file or fill the form and click 'Predict' to see results.")

# Function to call API for one employee's data and get prediction
def call_api_for_row(row_data_dict, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    api_payload = {"data": [row_data_dict]}  # Wrap single row inside 'data' key

    body = json.dumps(api_payload).encode("utf-8")
    req = urllib.request.Request(API_URL, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result_bytes = response.read()
        raw_response_string = result_bytes.decode("utf-8")

        # Decode nested JSON responses
        intermediate_string = json.loads(raw_response_string)
        output = json.loads(intermediate_string)

        # Default values in case API format is unexpected
        prediction_label = "N/A"
        probability_active = 0.0
        probability_terminated = 0.0

        if "predictions" in output and isinstance(output["predictions"], list) and len(output["predictions"]) > 0:
            prediction_label = output["predictions"][0]

        if "probabilities" in output and isinstance(output["probabilities"], list) and len(output["probabilities"]) > 0:
            if isinstance(output["probabilities"][0], list) and len(output["probabilities"][0]) >= 2:
                probability_active = output["probabilities"][0][0]
                probability_terminated = output["probabilities"][0][1]

        return prediction_label, probability_active, probability_terminated

    # Graceful handling of various error types
    except urllib.error.HTTPError as error:
        st.error(f"API request failed for a row with status code: {error.code}. Response: {error.read().decode('utf8', 'ignore')}")
        return "ERROR", 0.0, 0.0
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from API for a row: {e}. Raw response: {raw_response_string}")
        return "ERROR", 0.0, 0.0
    except Exception as e:
        st.error(f"An unexpected error occurred for a row: {e}")
        st.exception(e)
        return "ERROR", 0.0, 0.0



# --- Function to process uploaded file and run predictions ---
def run_batch_predictions(uploaded_file_content, api_key_val, required_api_columns):
    df_full = pd.read_excel(BytesIO(uploaded_file_content))

    # Validate required columns exist in uploaded file
    missing_cols = [col for col in required_api_columns if col not in df_full.columns]
    if missing_cols:
        st.error(f"Error: The uploaded file is missing required columns: {', '.join(missing_cols)}.")
        return None  # Stop further processing if required columns are missing

    # Prepare data for API by selecting only required columns
    df_processed_for_api = df_full[required_api_columns].copy()

    # Ensure numeric fields are properly typed
    for col in ['Hourly comp', 'Age']:
        if col in df_processed_for_api.columns:
            df_processed_for_api[col] = pd.to_numeric(df_processed_for_api[col], errors='coerce').fillna(0)
            if col == 'Age':
                df_processed_for_api[col] = df_processed_for_api[col].astype(int)
            else:
                df_processed_for_api[col] = df_processed_for_api[col].astype(float)

    # Convert datetime fields to proper string format for API (if present)
    for col in ['Hire Date', 'Term date']:
        if col in df_processed_for_api.columns:
            df_processed_for_api[col] = pd.to_datetime(df_processed_for_api[col], errors='coerce')
            df_processed_for_api[col] = df_processed_for_api[col].dt.strftime('%Y-%m-%d').replace({np.nan: None, 'NaT': None})

    # Build list of records (dicts) for API call
    records_to_send = []
    for _, row in df_processed_for_api.iterrows():
        record_dict = {}
        for key, value in row.items():
            record_dict[key] = None if pd.isna(value) or value == '' else value
        records_to_send.append(record_dict)

    # Initialize columns to store predictions
    df_full["Predicted Status"] = ""
    df_full["Probability (Active)"] = 0.0
    df_full["Probability (Terminated)"] = 0.0

    # --- Progress bar setup ---
    st.info("Beginning batch prediction.")
    my_bar = st.progress(0, text="Processing records...")

    # Call API for each record and update prediction results
    for i, record_dict in enumerate(records_to_send):
        prediction, prob_active, prob_terminated = call_api_for_row(record_dict, api_key_val)
        df_full.loc[i, "Predicted Status"] = prediction
        df_full.loc[i, "Probability (Active)"] = prob_active * 100
        df_full.loc[i, "Probability (Terminated)"] = prob_terminated * 100

        percent_complete = (i + 1) / len(records_to_send)
        my_bar.progress(percent_complete, text=f"Processing record {i+1}/{len(records_to_send)}...")

    my_bar.empty()  # Clear progress bar
    st.success("Batch prediction complete!")

    # Add derived columns for visual analysis
    df_full['Predicted Attrition Probability'] = df_full['Probability (Terminated)']
    df_full['Attrition Risk Level'] = pd.cut(
        df_full['Predicted Attrition Probability'],
        bins=[0, 49.99, 75, 100],
        labels=['Low Risk (<50%)', 'Medium Risk (50-75%)', 'High Risk (>75%)'],
        right=True
    )

    return df_full



# --- Handle Batch Upload ---
if uploaded_file:
    with col_output:
        st.subheader("Processing Excel File...")

        # Generate a hash of file content to enable caching
        file_content_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        # Check if cached predictions exist for this exact file
        if "batch_predictions_cache" not in st.session_state or \
           st.session_state.batch_predictions_cache.get("file_hash") != file_content_hash:

            # New file detected – run predictions
            try:
                df_with_predictions = run_batch_predictions(
                    uploaded_file.getvalue(),
                    API_KEY,
                    REQUIRED_API_COLUMNS
                )
                if df_with_predictions is not None:
                    st.session_state.batch_predictions_cache = {
                        "file_hash": file_content_hash,
                        "df": df_with_predictions.copy()
                    }
                else:
                    st.stop()
            except Exception as e:
                st.error(f"Error processing Excel file or during prediction: {e}")
                st.exception(e)
                st.stop()
        else:
            # Load from cache
            st.info("Loading predictions from cache.")
            df_with_predictions = st.session_state.batch_predictions_cache["df"]

        # --- Display Results ---
        if df_with_predictions is not None:
            st.subheader("Batch Prediction Results Table")
            st.dataframe(df_with_predictions[DISPLAY_COLUMNS])

            # CSV download option
            csv_data = df_with_predictions[DISPLAY_COLUMNS].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="attrition_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Interactive Visualizations of Attrition Risk")

            # --- Scatter Plot: Hourly comp vs Age ---
            feature1, feature2 = 'Hourly comp', 'Age'
            if feature1 in df_with_predictions.columns and feature2 in df_with_predictions.columns:
                filtered_df = df_with_predictions[
                    (df_with_predictions[feature1] >= 5) & (df_with_predictions[feature1] <= 300)
                ].copy()

                color_map = {
                    'Low Risk (<50%)': 'green',
                    'Medium Risk (50-75%)': 'gold',
                    'High Risk (>75%)': 'red'
                }

                if not filtered_df.empty:
                    fig_scatter = px.scatter(
                        filtered_df,
                        x=feature1,
                        y=feature2,
                        color='Attrition Risk Level',
                        color_discrete_map=color_map,
                        hover_data=[feature1, feature2, 'Predicted Attrition Probability'],
                        title='Employee Attrition Risk Scatter Plot',
                        labels={feature1: feature1, feature2: feature2}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                else:
                    st.warning(f"No data to plot after filtering '{feature1}' between $5 and $300.")
            else:
                st.warning(f"Missing numerical columns: '{feature1}' or '{feature2}'")

            # --- Bar Plot: Average Attrition by State ---
            if 'State' in df_with_predictions.columns:
                state_risk = df_with_predictions.groupby('State')['Predicted Attrition Probability'].mean().reset_index()
                state_risk = state_risk.sort_values(by='Predicted Attrition Probability', ascending=False)

                fig_bar = px.bar(
                    state_risk,
                    x='State',
                    y='Predicted Attrition Probability',
                    color='Predicted Attrition Probability',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Average Predicted Attrition Risk by State',
                    hover_data={'Predicted Attrition Probability': ':.2'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Column 'State' not found in data.")

            # --- Bar Plot: Average Attrition by Home Department ---
            if 'Home Department' in df_with_predictions.columns:
                dept_risk = df_with_predictions.groupby('Home Department')['Predicted Attrition Probability'].mean().reset_index()
                dept_risk = dept_risk.sort_values(by='Predicted Attrition Probability', ascending=False)

                fig_dept_bar = px.bar(
                    dept_risk,
                    x='Home Department',
                    y='Predicted Attrition Probability',
                    color='Predicted Attrition Probability',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Average Predicted Attrition Risk by Home Department',
                    hover_data={'Predicted Attrition Probability': ':.2'}
                )
                fig_dept_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_dept_bar, use_container_width=True)
            else:
                st.warning("Column 'Home Department' not found in data.")



# --- Handle Single Prediction ---
if submit_single:
    with col_output:
        st.subheader("Single Prediction Output")

        # Prepare input dictionary
        single_row_data = {
            "Gender": gender,
            "Marital status": marital_status,
            "Hourly comp": hourly_comp,
            "Race": race,
            "State": state,
            "Age": age,
            "Hire Date": str(hire_date),
            "Employee type": employee_type,
            "Rehire": rehire,
            "Home Department": home_dept
        }

        # Call API and interpret results
        prediction_label, probability_active, probability_terminated = call_api_for_row(single_row_data, API_KEY)
        perc_active = probability_active * 100
        perc_term = probability_terminated * 100

        if prediction_label != "ERROR":
            display_status = "Stay" if prediction_label == "A" else "Leave"
            st.write(f"**Predicted Employee Status:** `{display_status}`")

            st.write("**Class Probabilities:**")
            st.markdown(f"""
            - Probability of Staying: **{perc_active:.2f}%**  
            - Probability of Leaving: **{perc_term:.2f}%**
            """)
        else:
            st.error("Failed to get prediction for the single employee.")

        st.subheader("Input Data Sent to API")
        st.json(single_row_data)

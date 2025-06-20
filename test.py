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
API_URL = "https://attrition-pred-v2-1.eastus2.inference.ml.azure.com/score" 
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
    "XGBoost Prediction",
    "XGBoost Prob. Leave",       # Overall/Current Tenure XGBoost Prob
    "XGBoost Prob. Leave (3mo)", # NEW: Adjusted XGBoost Prob for 3 months
    "XGBoost Prob. Leave (6mo)", # NEW: Adjusted XGBoost Prob for 6 months
    "XGBoost Prob. Leave (12mo)",# NEW: Adjusted XGBoost Prob for 12 months
    "Cox Prob. Leave (3mo)",
    "Cox Prob. Leave (6mo)",
    "Cox Prob. Leave (12mo)",
    "Attrition Risk Level",      # Derived from XGBoost Prob. Leave
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
        marital_status = st.selectbox("Marital Status", ["S", "M", "D", "W", "O"], key="single_marital", help="S: Single, M: Married, D: Divorced, W: Widowed, O: Other")
        hourly_comp = st.number_input("Hourly Compensation", min_value=0.0, step=0.5, value=5.0, key="single_hourly")
        race = st.selectbox("Race", ["B","H","T","W","X", "A", "I", "D", "O", "P"], key="single_race", help="Common racial codes. Backend performs mapping.")
        state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                                        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                                        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                                        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                                        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                                        "DC" ], key="single_state")
        age = st.number_input("Age", min_value=16, max_value=100, step=1, value=20, key="single_age")
        
        six_months_ago = date.today() - timedelta(days=6*30) 
        hire_date = st.date_input("Hire Date", value=six_months_ago, key="single_hire_date")

        employee_type = st.selectbox("Employee Type", ["Full time", "Part time", "Temporary"], key="single_emp_type")
        rehire = st.selectbox("Rehire", ["Yes", "No"], key="single_rehire")
        home_dept = st.text_input("Home Department", key="single_home_dept")

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

    api_payload = {"data": [row_data_dict]}
    body = json.dumps(api_payload).encode("utf-8")
    req = urllib.request.Request(API_URL, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result_bytes = response.read()
        raw_response_string = result_bytes.decode("utf-8")

        # # --- DEBUGGING LINE: Raw API Response ---
        # st.sidebar.write("--- API Response Debug (call_api_for_row) ---")
        # st.sidebar.write(f"Raw response (first 500 chars): {raw_response_string[:500]}...") 
        # # --- END DEBUGGING LINE ---

        try:
            intermediate_dict = json.loads(raw_response_string)
            if isinstance(intermediate_dict, str):
                output = json.loads(intermediate_dict)
            else:
                output = intermediate_dict
        except json.JSONDecodeError:
            output = json.loads(raw_response_string)

        # # --- DEBUGGING LINE: Parsed API Output ---
        # st.sidebar.write(f"Parsed Output Dict keys: {output.keys()}")
        # if "predictions" in output and isinstance(output["predictions"], list) and len(output["predictions"]) > 0:
        #     st.sidebar.write(f"First Prediction Dict keys: {output['predictions'][0].keys()}")
        #     st.sidebar.json(output['predictions'][0]) # Show full content of first prediction
        # else:
        #     st.sidebar.write("No 'predictions' key or empty list in API response.")
        # # --- END DEBUGGING LINE ---

        # Initialize all prediction results with None
        xgb_prediction = None
        xgb_prob_leave = None
        xgb_prob_leave_3mo = None
        xgb_prob_leave_6mo = None
        xgb_prob_leave_12mo = None
        cox_leave_3mo = None
        cox_leave_6mo = None
        cox_leave_12mo = None

        if "predictions" in output and isinstance(output["predictions"], list) and len(output["predictions"]) > 0:
            first_prediction_dict = output["predictions"][0] 
            
            xgb_prediction = first_prediction_dict.get("xgb_prediction")
            xgb_prob_leave = first_prediction_dict.get("xgb_prob_leave")
            
            xgb_prob_leave_3mo = first_prediction_dict.get("xgb_prob_leave_3mo")
            xgb_prob_leave_6mo = first_prediction_dict.get("xgb_prob_leave_6mo")
            xgb_prob_leave_12mo = first_prediction_dict.get("xgb_prob_leave_12mo")

            cox_leave_3mo = first_prediction_dict.get("cox_leave_3mo")
            cox_leave_6mo = first_prediction_dict.get("cox_leave_6mo")
            cox_leave_12mo = first_prediction_dict.get("cox_leave_12mo")

        return {
            "xgb_prediction": xgb_prediction,
            "xgb_prob_leave": xgb_prob_leave,
            "xgb_prob_leave_3mo": xgb_prob_leave_3mo,
            "xgb_prob_leave_6mo": xgb_prob_leave_6mo,
            "xgb_prob_leave_12mo": xgb_prob_leave_12mo,
            "cox_leave_3mo": cox_leave_3mo,
            "cox_leave_6mo": cox_leave_6mo,
            "cox_leave_12mo": cox_leave_12mo
        }

    except urllib.error.HTTPError as error:
        st.error(f"API request failed for a row with status code: {error.code}. Response: {error.read().decode('utf8', 'ignore')}")
        # # --- DEBUGGING LINE: Error return ---
        # st.sidebar.error(f"API HTTPError: {error.code} - {error.read().decode('utf8', 'ignore')}")
        # # --- END DEBUGGING LINE ---
        return {
            "xgb_prediction": "ERROR",
            "xgb_prob_leave": np.nan, 
            "xgb_prob_leave_3mo": np.nan,
            "xgb_prob_leave_6mo": np.nan,
            "xgb_prob_leave_12mo": np.nan,
            "cox_leave_3mo": np.nan,
            "cox_leave_6mo": np.nan,
            "cox_leave_12mo": np.nan
        }
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from API for a row: {e}. Raw response: {raw_response_string}")
        # # --- DEBUGGING LINE: Error return ---
        # st.sidebar.error(f"API JSONDecodeError: {e}. Raw: {raw_response_string[:500]}...")
        # # --- END DEBUGGING LINE ---
        return {
            "xgb_prediction": "ERROR",
            "xgb_prob_leave": np.nan,
            "xgb_prob_leave_3mo": np.nan,
            "xgb_prob_leave_6mo": np.nan,
            "xgb_prob_leave_12mo": np.nan,
            "cox_leave_3mo": np.nan,
            "cox_leave_6mo": np.nan,
            "cox_leave_12mo": np.nan
        }
    except Exception as e:
        st.error(f"An unexpected error occurred for a row: {e}")
        st.exception(e)
        # # --- DEBUGGING LINE: Error return ---
        # st.sidebar.error(f"API General Error: {e}")
        # # --- END DEBUGGING LINE ---
        return {
            "xgb_prediction": "ERROR",
            "xgb_prob_leave": np.nan,
            "xgb_prob_leave_3mo": np.nan,
            "xgb_prob_leave_6mo": np.nan,
            "xgb_prob_leave_12mo": np.nan,
            "cox_leave_3mo": np.nan,
            "cox_leave_6mo": np.nan,
            "cox_leave_12mo": np.nan
        }


# --- Function to process uploaded file and run predictions ---
def run_batch_predictions(uploaded_file_content, api_key_val, required_api_columns):
    df_full = pd.read_excel(BytesIO(uploaded_file_content))

    # Validate required columns exist in uploaded file
    missing_cols = [col for col in required_api_columns if col not in df_full.columns]
    if missing_cols:
        st.error(f"Error: The uploaded file is missing required columns: {', '.join(missing_cols)}.")
        return None 

    if 'Term date' not in df_full.columns:
        df_full['Term date'] = None
    else:
        df_full['Term date'] = None 

    df_processed_for_api = df_full[required_api_columns].copy()

    for col in ['Hourly comp', 'Age']:
        if col in df_processed_for_api.columns:
            df_processed_for_api[col] = pd.to_numeric(df_processed_for_api[col], errors='coerce').fillna(0)
            if col == 'Age':
                df_processed_for_api[col] = df_processed_for_api[col].astype(int)
            else:
                df_processed_for_api[col] = df_processed_for_api[col].astype(float)

    for col in ['Hire Date', 'Term date']:
        if col in df_processed_for_api.columns:
            df_processed_for_api[col] = pd.to_datetime(df_processed_for_api[col], errors='coerce')
            df_processed_for_api[col] = df_processed_for_api[col].dt.strftime('%Y-%m-%d').replace({np.nan: None, 'NaT': None})

    records_to_send = []
    for _, row in df_processed_for_api.iterrows():
        record_dict = {}
        for key, value in row.items():
            record_dict[key] = None if pd.isna(value) or value == '' else value
        records_to_send.append(record_dict)

    # Initialize new columns to store predictions
    df_full["XGBoost Prediction"] = ""
    df_full["XGBoost Prob. Leave"] = np.nan
    df_full["XGBoost Prob. Leave (3mo)"] = np.nan
    df_full["XGBoost Prob. Leave (6mo)"] = np.nan
    df_full["XGBoost Prob. Leave (12mo)"] = np.nan
    df_full["Cox Prob. Leave (3mo)"] = np.nan
    df_full["Cox Prob. Leave (6mo)"] = np.nan
    df_full["Cox Prob. Leave (12mo)"] = np.nan

    # # --- DEBUGGING LINE: Confirming initial columns ---
    # st.sidebar.write("--- Batch Prediction Init Debug ---")
    # st.sidebar.write(f"df_full columns after initialization: {df_full.columns.tolist()}")
    # # --- END DEBUGGING LINE ---

    st.info("Beginning batch prediction. This may take a while for large files...")
    my_bar = st.progress(0, text="Processing records...")

    for i, record_dict in enumerate(records_to_send):
        prediction_results = call_api_for_row(record_dict, api_key_val)
        
        xgb_prediction = prediction_results.get("xgb_prediction")
        xgb_prob_leave = prediction_results.get("xgb_prob_leave")
        xgb_3mo = prediction_results.get("xgb_prob_leave_3mo")
        xgb_6mo = prediction_results.get("xgb_prob_leave_6mo")
        xgb_12mo = prediction_results.get("xgb_prob_leave_12mo")
        cox_3mo = prediction_results.get("cox_leave_3mo")
        cox_6mo = prediction_results.get("cox_leave_6mo")
        cox_12mo = prediction_results.get("cox_leave_12mo")

        df_full.loc[i, "XGBoost Prob. Leave"] = xgb_prob_leave * 100 if xgb_prob_leave is not None else np.nan
        df_full.loc[i, "XGBoost Prob. Leave (3mo)"] = xgb_3mo * 100 if xgb_3mo is not None else np.nan
        df_full.loc[i, "XGBoost Prob. Leave (6mo)"] = xgb_6mo * 100 if xgb_6mo is not None else np.nan
        df_full.loc[i, "XGBoost Prob. Leave (12mo)"] = xgb_12mo * 100 if xgb_12mo is not None else np.nan
        df_full.loc[i, "Cox Prob. Leave (3mo)"] = cox_3mo * 100 if cox_3mo is not None else np.nan
        df_full.loc[i, "Cox Prob. Leave (6mo)"] = cox_6mo * 100 if cox_6mo is not None else np.nan
        df_full.loc[i, "Cox Prob. Leave (12mo)"] = cox_12mo * 100 if cox_12mo is not None else np.nan

        percent_complete = (i + 1) / len(records_to_send)
        my_bar.progress(percent_complete, text=f"Processing record {i+1}/{len(records_to_send)}...")

    my_bar.empty()
    st.success("Batch prediction complete!")

    df_full['Predicted Attrition Probability'] = df_full['XGBoost Prob. Leave'] 
    df_full['Attrition Risk Level'] = pd.cut(
        df_full['Predicted Attrition Probability'],
        bins=[0, 49.99, 75, 100],
        labels=['Low Risk (<50%)', 'Medium Risk (50-75%)', 'High Risk (>75%)'],
        right=True,
        include_lowest=True
    )
    
    # # --- DEBUGGING LINE FOR BATCH DF before return ---
    # st.sidebar.write("--- Batch DF Columns before return ---")
    # st.sidebar.write(f"df_full columns before returning: {df_full.columns.tolist()}")
    # # --- END DEBUGGING LINE ---

    return df_full


# --- Handle Batch Upload ---
if uploaded_file:
    with col_output:
        st.subheader("Processing Excel File...")

        file_content_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        if "batch_predictions_cache" not in st.session_state or \
           st.session_state.batch_predictions_cache.get("file_hash") != file_content_hash:

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
            st.info("Loading predictions from cache.")
            df_with_predictions = st.session_state.batch_predictions_cache["df"]

        # --- Display Results ---
        if df_with_predictions is not None:
            # # --- DEBUGGING LINE: Final DF Columns ---
            # st.sidebar.write("--- Final DF Columns for Display ---")
            # st.sidebar.write(f"df_with_predictions columns at display: {df_with_predictions.columns.tolist()}")
            # st.sidebar.write(f"DISPLAY_COLUMNS needed: {DISPLAY_COLUMNS}")
            # # --- END DEBUGGING LINE ---

            st.subheader("Batch Prediction Results Table")
            st.dataframe(df_with_predictions[DISPLAY_COLUMNS]) # This is the line causing the error

            csv_data = df_with_predictions[DISPLAY_COLUMNS].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="attrition_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Interactive Visualizations of Attrition Risk")

            feature1, feature2 = 'Hourly comp', 'Age'
            if feature1 in df_with_predictions.columns and feature2 in df_with_predictions.columns:
                filtered_df = df_with_predictions[
                    (df_with_predictions[feature1] >= 5) & (df_with_predictions[feature1] <= 300) &
                    (df_with_predictions[feature2] >= 18) & (df_with_predictions[feature2] <= 70)
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
                        title='Employee Attrition Risk Scatter Plot (Hourly Comp vs Age)',
                        labels={feature1: feature1, feature2: feature2}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                else:
                    st.warning(f"No data to plot after filtering '{feature1}' between $5 and $300 and '{feature2}' between 18 and 70.")
            else:
                st.warning(f"Missing numerical columns for scatter plot: '{feature1}' or '{feature2}'")

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
                    hover_data={'Predicted Attrition Probability': ':.2f'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Column 'State' not found in data for plotting.")

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
                    hover_data={'Predicted Attrition Probability': ':.2f'}
                )
                fig_dept_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_dept_bar, use_container_width=True)
            else:
                st.warning("Column 'Home Department' not found in data for plotting.")


# --- Handle Single Prediction ---
if submit_single:
    with col_output:
        st.subheader("Single Prediction Output")

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

        prediction_results = call_api_for_row(single_row_data, API_KEY)
        
        xgb_prediction = prediction_results.get("xgb_prediction")
        xgb_prob_leave = prediction_results.get("xgb_prob_leave")
        xgb_3mo = prediction_results.get("xgb_prob_leave_3mo")
        xgb_6mo = prediction_results.get("xgb_prob_leave_6mo")
        xgb_12mo = prediction_results.get("xgb_prob_leave_12mo")
        cox_3mo = prediction_results.get("cox_leave_3mo")
        cox_6mo = prediction_results.get("cox_leave_6mo")
        cox_12mo = prediction_results.get("cox_leave_12mo")

        if xgb_prediction is not None and xgb_prediction != "ERROR":
            # display_status = "Terminated (Leave)" if xgb_prediction == 1 else "Active (Stay)"
            # st.write(f"**XGBoost Predicted Employee Status:** `{display_status}`")

            st.write("**Probabilities:**")
            if xgb_prob_leave is not None:
                st.markdown(f"- **XGBoost Probability of Leaving (Current Tenure):** **{xgb_prob_leave * 100:.2f}%**")
            if xgb_3mo is not None:
                st.markdown(f"- **XGBoost Probability of Leaving (3 months from now):** **{xgb_3mo * 100:.2f}%**")
            if xgb_6mo is not None:
                st.markdown(f"- **XGBoost Probability of Leaving (6 months from now):** **{xgb_6mo * 100:.2f}%**")
            if xgb_12mo is not None:
                st.markdown(f"- **XGBoost Probability of Leaving (12 months from now):** **{xgb_12mo * 100:.2f}%**")
            st.markdown(f"""
            - **Cox PH Probability of Leaving (3 months):** **{cox_3mo * 100:.2f}%**
            - **Cox PH Probability of Leaving (6 months):** **{cox_6mo * 100:.2f}%**
            - **Cox PH Probability of Leaving (12 months):** **{cox_12mo * 100:.2f}%**
            """)
        else:
            st.error("Failed to get prediction for the single employee.")

        st.subheader("Input Data Sent to API")
        st.json(single_row_data)

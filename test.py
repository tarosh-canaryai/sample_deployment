import streamlit as st
import json
import urllib.request
import urllib.error
import pandas as pd
from datetime import date
from io import BytesIO
import plotly.express as px
import numpy as np

API_URL = "https://attrition-pred-v1-debug-score.eastus2.inference.ml.azure.com/score"
API_KEY = st.secrets["API_KEY"]

REQUIRED_API_COLUMNS = [
    "Term date",
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

st.set_page_config(layout="wide")
st.title("Employee Attrition Predictor ")

# Create two columns for layout
col_input, col_output = st.columns(2)

with col_input:
    st.header("Upload Employee Data (Excel)")
    st.write("Or use the manual form below for single predictions.")

    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help=f"Upload an Excel file containing employee data for batch prediction. "
             f"Ensure your Excel sheet has columns named: {', '.join(REQUIRED_API_COLUMNS)}."
    )

    st.markdown("---")
    st.header("Or Enter Single Employee Details")
    with st.form("attrition_form_single"):
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="single_gender")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"], key="single_marital")
        hourly_comp = st.number_input("Hourly Compensation", min_value=0.0, step=0.5, key="single_hourly")
        race = st.selectbox("Race", ["Asian", "Black", "White", "Hispanic", "Other"], key="single_race")
        state = st.selectbox("State", ["NY", "CA", "TX", "FL", "IL", "Other"], key="single_state")
        age = st.number_input("Age", min_value=16, max_value=100, step=1, key="single_age")
        hire_date = st.date_input("Hire Date", value=date.today(), key="single_hire_date")
        employee_type = st.selectbox("Employee Type", ["Full-time", "Part-time", "Temporary"], key="single_emp_type")
        rehire = st.selectbox("Rehire", ["Yes", "No"], key="single_rehire")
        home_dept = st.text_input("Home Department", key="single_home_dept")
        term_date = st.text_input("Termination Date (YYYY-MM-DD or leave blank)", placeholder="Optional", key="single_term_date")

        submit_single = st.form_submit_button("Predict Single Employee")


with col_output:
    st.header("Prediction Results")
    results_placeholder = st.empty() # Placeholder for dynamic results

    if not uploaded_file and not submit_single:
        results_placeholder.info("Upload an Excel file or fill the form and click 'Predict' to see results.")


def call_api_for_row(row_data_dict):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    api_payload = {"data": [row_data_dict]}

    body = json.dumps(api_payload).encode("utf-8")
    req = urllib.request.Request(API_URL, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result_bytes = response.read()
        raw_response_string = result_bytes.decode("utf-8")

        intermediate_string = json.loads(raw_response_string)
        output = json.loads(intermediate_string)

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

    except urllib.error.HTTPError as error:
        st.error(f"API request failed for a row with status code: {error.code}. Response: {error.read().decode('utf8', 'ignore')}")
        return "ERROR", 0.0, 0.0
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from API for a row: {e}. Raw response: {raw_response_string}")
    except Exception as e:
        st.error(f"An unexpected error occurred for a row: {e}")
        st.exception(e)
        return "ERROR", 0.0, 0.0


# --- Handle Batch Upload ---
if uploaded_file:
    with col_output:
        st.subheader("Processing Excel File...")

        try:
            df_full = pd.read_excel(BytesIO(uploaded_file.getvalue()))
            st.success(f"Successfully loaded {len(df_full)} rows from '{uploaded_file.name}'.")

            missing_cols = [col for col in REQUIRED_API_COLUMNS if col not in df_full.columns]
            if missing_cols:
                st.error(f"Error: The uploaded file is missing required columns: {', '.join(missing_cols)}. "
                         f"Please ensure your Excel has these exact column names.")
                st.stop() # Stop execution if critical columns are missing

            # Create a copy with only the required columns for API processing
            df_processed_for_api = df_full[REQUIRED_API_COLUMNS].copy()

            # --- Type Conversions and Handling NaN/NaT for API  ---
            for col in ['Hourly comp', 'Age']:
                if col in df_processed_for_api.columns:
                    df_processed_for_api[col] = pd.to_numeric(df_processed_for_api[col], errors='coerce').fillna(0)
                    if col == 'Age':
                        df_processed_for_api[col] = df_processed_for_api[col].astype(int)
                    else: # Hourly comp
                        df_processed_for_api[col] = df_processed_for_api[col].astype(float)

            for col in ['Hire Date', 'Term date']:
                if col in df_processed_for_api.columns:
                    df_processed_for_api[col] = pd.to_datetime(df_processed_for_api[col], errors='coerce')
                    df_processed_for_api[col] = df_processed_for_api[col].dt.strftime('%Y-%m-%d').replace({np.nan: None, 'NaT': None})

            records_to_send = []
            for _, row in df_processed_for_api.iterrows():
                record_dict = {}
                for key, value in row.items():
                    if pd.isna(value) or value == '':
                        record_dict[key] = None
                    else:
                        record_dict[key] = value
                records_to_send.append(record_dict)

            st.info("Beginning batch prediction.")

            # Prepare columns for prediction results
            df_full["Predicted Status"] = ""
            df_full["Probability (Active)"] = 0.0
            df_full["Probability (Terminated)"] = 0.0


            progress_text = "Sending data to API and getting predictions."
            my_bar = st.progress(0, text=progress_text)

            for i, record_dict in enumerate(records_to_send):
                prediction, prob_active, prob_terminated = call_api_for_row(record_dict)

                df_full.loc[i, "Predicted Status"] = prediction
                df_full.loc[i, "Probability (Active)"] = prob_active
                df_full.loc[i, "Probability (Terminated)"] = prob_terminated

                progress = (i + 1) / len(records_to_send)
                my_bar.progress(progress, text=f"{progress_text} ({int(progress*100)}%)")

            my_bar.empty()
            st.success("Batch prediction complete!")

            # Define the columns for the final display and download
            DISPLAY_COLUMNS = REQUIRED_API_COLUMNS + [
                "Predicted Status",
                "Probability (Active)",
                "Probability (Terminated)"
            ]
            
            # Create the DataFrame for display and download
            df_display_download = df_full[DISPLAY_COLUMNS].copy()

            st.subheader("Batch Prediction Results Table")
            st.dataframe(df_display_download) # Display the specific subset

            csv = df_display_download.to_csv(index=False).encode('utf-8') # Export the specific subset
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="attrition_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Interactive Visualizations of Attrition Risk")

            # --- Scatter Plot of Employee Attrition Risk (Plotly) ---
            # Use df_full for plotting as it now contains prediction columns and we need to derive risk level
            df_full['Predicted Attrition Probability'] = df_full['Probability (Terminated)']

            # Create a categorical risk level for coloring and legend
            df_full['Attrition Risk Level'] = pd.cut(
                df_full['Predicted Attrition Probability'],
                bins=[0, 0.4999, 0.75, 1.0],
                labels=['Low Risk (<50%)', 'Medium Risk (50-75%)', 'High Risk (>75%)'],
                right=True # Ensure 0.50 falls into medium risk
            )

            color_map = {
                'Low Risk (<50%)': 'green',
                'Medium Risk (50-75%)': 'gold',
                'High Risk (>75%)': 'red'
            }

            feature1 = 'Hourly comp'
            feature2 = 'Age'

            if feature1 in df_full.columns and feature2 in df_full.columns:
                # Filter for Hourly comp between 5 and 300
                filtered_results_df_for_plot = df_full[
                    (df_full[feature1] >= 5) & (df_full[feature1] <= 300)
                ].copy()

                if not filtered_results_df_for_plot.empty:
                    fig_scatter = px.scatter(
                        filtered_results_df_for_plot, # Use the filtered DataFrame for plotting
                        x=feature1,
                        y=feature2,
                        color='Attrition Risk Level',
                        color_discrete_map=color_map,
                        # Adjusted hover_data to include relevant info for the plot
                        hover_data=[feature1, feature2, 'Predicted Attrition Probability', 'Attrition Risk Level'],
                        title=f'Employee Attrition Risk Scatter Plot ({feature1} between $5 and $300)',
                        labels={
                            feature1: feature1,
                            feature2: feature2,
                            'Predicted Attrition Probability': 'Predicted Attrition Probability'
                        }
                    )
                    fig_scatter.update_traces(
                        hovertemplate=f"<b>{feature1}:</b> %{{x}}<br><b>{feature2}:</b> %{{y}}<br><b>Predicted Attrition Probability:</b> %{{customdata[0]:.2%}}<br><extra></extra>",
                        customdata=np.stack((filtered_results_df_for_plot['Predicted Attrition Probability'],), axis=-1)
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning(f"No data to plot for scatter plot after filtering '{feature1}' between $5 and $300.")
            else:
                st.warning(f"Cannot generate scatter plot: Features '{feature1}' or '{feature2}' not found or are not numerical.")

            # --- Predicted Attrition Risk by State (Bar Chart - Plotly) ---
            if 'State' in df_full.columns: # Use df_full for this as well
                state_attrition_risk = df_full.groupby('State')['Predicted Attrition Probability'].mean().reset_index()
                state_attrition_risk = state_attrition_risk.sort_values(by='Predicted Attrition Probability', ascending=False)

                fig_bar = px.bar(
                    state_attrition_risk,
                    x='State',
                    y='Predicted Attrition Probability',
                    color='Predicted Attrition Probability',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Average Predicted Attrition Risk by State',
                    labels={
                        'Predicted Attrition Probability': 'Average Predicted Attrition Probability'
                    },
                    hover_data={'Predicted Attrition Probability': ':.2%'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.warning("Cannot analyze attrition by state: 'State' column not found in the uploaded data.")

        except pd.errors.EmptyDataError:
            st.error("The uploaded Excel file is empty.")
        except Exception as e:
            st.error(f"Error reading or processing Excel file: {e}")
            st.exception(e)


# --- Handle Single Prediction ---
if submit_single:
    with col_output:
        st.subheader("Single Prediction Output")

        single_row_data = {
            "Term date": term_date if term_date else None,
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

        prediction_label, probability_active, probability_terminated = call_api_for_row(single_row_data)

        if prediction_label != "ERROR":
            st.write(f"**Predicted Employee Status:** `{prediction_label}`")
            st.write("**Class Probabilities:**")
            st.markdown(f"""
            - `A` (Active): **{probability_active:.2%}**
            - `T` (Terminated): **{probability_terminated:.2%}**
            """)
        else:
            st.error("Failed to get prediction for the single employee. See errors above.")

        st.subheader("Input Data Sent to API")
        st.json(single_row_data)
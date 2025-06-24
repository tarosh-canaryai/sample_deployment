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

# Base columns to always show in final output table
DISPLAY_COLUMNS_BASE_STATIC = REQUIRED_API_COLUMNS.copy()

st.set_page_config(layout="wide")
st.title("Employee Attrition Predictor")

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

        try:
            intermediate_dict = json.loads(raw_response_string)
            if isinstance(intermediate_dict, str):
                output = json.loads(intermediate_dict)
            else:
                output = intermediate_dict
        except json.JSONDecodeError:
            output = json.loads(raw_response_string)

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
        # If 'Term date' exists, ensure it's handled as 'None' for prediction for current employees
        # and kept for historical analysis if needed later (not currently used for API input)
        df_full['Term date'] = None

    df_processed_for_api = df_full[required_api_columns].copy()

    for col in ['Hourly comp', 'Age']:
        if col in df_processed_for_api.columns:
            df_processed_for_api[col] = pd.to_numeric(df_processed_for_api[col], errors='coerce').fillna(0)
            if col == 'Age':
                df_processed_for_api[col] = df_processed_for_api[col].astype(int)
            else:
                df_processed_for_api[col] = df_processed_for_api[col].astype(float)

    for col in ['Hire Date']: # Only Hire Date is consistently used as input
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
    df_full["XGBoost Prob. Leave"] = np.nan # Kept internally for risk level calculation
    df_full["Consolidated Prob. Leave (3mo)"] = np.nan
    df_full["Consolidated Prob. Leave (6mo)"] = np.nan
    df_full["Consolidated Prob. Leave (12mo)"] = np.nan

    # Initialize new columns for dynamic Attrition Risk Level per timeframe
    df_full["Attrition Risk Level (3mo)"] = None
    df_full["Attrition Risk Level (6mo)"] = None
    df_full["Attrition Risk Level (12mo)"] = None


    st.info("Beginning batch prediction. This may take a while for large files...")
    my_bar = st.progress(0, text="Processing records...")

    for i, record_dict in enumerate(records_to_send):
        prediction_results = call_api_for_row(record_dict, api_key_val)

        xgb_prob_leave = prediction_results.get("xgb_prob_leave") # Primary XGB prob for overall risk
        xgb_3mo = prediction_results.get("xgb_prob_leave_3mo")
        xgb_6mo = prediction_results.get("xgb_prob_leave_6mo")
        xgb_12mo = prediction_results.get("xgb_prob_leave_12mo")
        cox_3mo = prediction_results.get("cox_leave_3mo")
        cox_6mo = prediction_results.get("cox_leave_6mo")
        cox_12mo = prediction_results.get("cox_leave_12mo")

        df_full.loc[i, "XGBoost Prob. Leave"] = xgb_prob_leave * 100 if xgb_prob_leave is not None else np.nan

        # Calculate Consolidated Probabilities
        consolidated_3mo_val = ((xgb_3mo if xgb_3mo is not None else 0) + (cox_3mo if cox_3mo is not None else 0)) / 2 * 100 \
            if xgb_3mo is not None or cox_3mo is not None else np.nan
        consolidated_6mo_val = ((xgb_6mo if xgb_6mo is not None else 0) + (cox_6mo if cox_6mo is not None else 0)) / 2 * 100 \
            if xgb_6mo is not None or cox_6mo is not None else np.nan
        consolidated_12mo_val = ((xgb_12mo if xgb_12mo is not None else 0) + (cox_12mo if cox_12mo is not None else 0)) / 2 * 100 \
            if xgb_12mo is not None or cox_12mo is not None else np.nan

        df_full.loc[i, "Consolidated Prob. Leave (3mo)"] = consolidated_3mo_val
        df_full.loc[i, "Consolidated Prob. Leave (6mo)"] = consolidated_6mo_val
        df_full.loc[i, "Consolidated Prob. Leave (12mo)"] = consolidated_12mo_val

        # Calculate Attrition Risk Level for each timeframe dynamically
        for prob_col, risk_col in [
            ("Consolidated Prob. Leave (3mo)", "Attrition Risk Level (3mo)"),
            ("Consolidated Prob. Leave (6mo)", "Attrition Risk Level (6mo)"),
            ("Consolidated Prob. Leave (12mo)", "Attrition Risk Level (12mo)")
        ]:
            if not pd.isna(df_full.loc[i, prob_col]):
                df_full.loc[i, risk_col] = pd.cut(
                    [df_full.loc[i, prob_col]],
                    bins=[0, 49.99, 75, 100],
                    labels=['Low Risk (<50%)', 'Medium Risk (50-75%)', 'High Risk (>75%)'],
                    right=True,
                    include_lowest=True
                )[0]

        percent_complete = (i + 1) / len(records_to_send)
        my_bar.progress(percent_complete, text=f"Processing record {i+1}/{len(records_to_send)}...")

    my_bar.empty()
    st.success("Batch prediction complete!")

    # 'Attrition Risk Level' column is no longer directly set here for the entire DataFrame,
    # as it will be selected dynamically for display.
    # The individual 'Attrition Risk Level (Xmo)' columns now hold the specific risk levels.

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
            st.subheader("Batch Prediction Results Table")

            # HR-Friendly Filters and Display
            timeframe_options = ["All", "3 months", "6 months", "12 months"]
            selected_timeframe = st.selectbox(
                "Select Attrition Timeframe",
                timeframe_options,
                index=timeframe_options.index("6 months"), # Default to 6 months
                key="batch_timeframe_selector"
            )

            # Determine which risk column and probability column to use based on selected_timeframe
            if selected_timeframe == "All":
                display_prob_columns = [
                    "Consolidated Prob. Leave (3mo)",
                    "Consolidated Prob. Leave (6mo)",
                    "Consolidated Prob. Leave (12mo)"
                ]
                # For sorting and plotting when 'All' is selected, we'll default to 6mo
                selected_risk_column_name = "Attrition Risk Level (6mo)"
                selected_prob_column_for_plots = "Consolidated Prob. Leave (6mo)"
            else:
                display_prob_columns = [f"Consolidated Prob. Leave ({selected_timeframe.replace(' months', 'mo')})"]
                selected_risk_column_name = f"Attrition Risk Level ({selected_timeframe.replace(' months', 'mo')})"
                selected_prob_column_for_plots = display_prob_columns[0]

            risk_filter_options = ["All", "Low Risk (<50%)", "Medium Risk (50-75%)", "High Risk (>75%)"]
            selected_risk_filter = st.selectbox(
                "Filter by Attrition Risk Level",
                risk_filter_options,
                index=0, # Default to 'All'
                key="batch_risk_filter_selector"
            )

            current_display_columns = DISPLAY_COLUMNS_BASE_STATIC + [selected_risk_column_name] + display_prob_columns

            # Filter data based on risk level, using the dynamically selected risk column
            filtered_df_for_display = df_with_predictions.copy()
            if selected_risk_filter != "All":
                filtered_df_for_display = filtered_df_for_display[
                    filtered_df_for_display[selected_risk_column_name] == selected_risk_filter
                ]

            if not filtered_df_for_display.empty:
                # Sort by the relevant probability column
                sort_column = selected_prob_column_for_plots
                if sort_column not in filtered_df_for_display.columns:
                    # Fallback if the chosen plot column isn't in filtered data, should not happen with current logic
                    sort_column = display_prob_columns[0] if display_prob_columns else 'Attrition Risk Level (6mo)' # Safest default
                    if sort_column not in filtered_df_for_display.columns:
                        sort_column = filtered_df_for_display.columns[0] # Fallback to first column

                # Rename the selected_risk_column_name for display in the dataframe for cleaner output
                df_to_display = filtered_df_for_display[current_display_columns].rename(columns={selected_risk_column_name: "Attrition Risk Level"})
                st.dataframe(df_to_display.sort_values(by=sort_column, ascending=False))

                # --- START Code for Overall Attrition Risk Distribution Summary ---
                if not filtered_df_for_display.empty:
                    st.markdown("---")
                    st.subheader("Overall Attrition Risk Distribution")

                    risk_counts = filtered_df_for_display[selected_risk_column_name].value_counts().reindex(
                        ['High Risk (>75%)', 'Medium Risk (50-75%)', 'Low Risk (<50%)']
                    ).fillna(0).astype(int)

                    total_employees_displayed = len(filtered_df_for_display)

                    if total_employees_displayed > 0:
                        st.markdown(
                            f"**Total Employees Displayed:** {total_employees_displayed}  \n"
                            f"**Filters Applied:** Timeframe = *{selected_timeframe}*, Risk Level = *{selected_risk_filter}*"
                        )

                        st.markdown("#### Risk Breakdown:")
                        for risk_level, count in risk_counts.items():
                            percentage = (count / total_employees_displayed) * 100
                            st.markdown(
                                f"- **{risk_level}**: {count} employees ({percentage:.1f}%)"
                            )

                        st.markdown(
                            f"*Risk levels are calculated using the **{selected_timeframe}** consolidated probability.*\n"
                        )
                    else:
                        st.info("No employees to analyze for overall risk distribution based on current filters.")


                # --- END Code for Overall Attrition Risk Distribution Summary ---

                csv_data = df_to_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Results for {selected_timeframe} as CSV",
                    data=csv_data,
                    file_name=f"attrition_predictions_{selected_timeframe.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No employees match the selected filters.")

            st.subheader("Interactive Visualizations of Attrition Risk")

            # Ensure plots use the filtered data and selected timeframe probability
            if not filtered_df_for_display.empty:
                feature1, feature2 = 'Hourly comp', 'Age'
                if feature1 in filtered_df_for_display.columns and feature2 in filtered_df_for_display.columns:
                    plot_df = filtered_df_for_display[
                        (filtered_df_for_display[feature1] >= 5) & (filtered_df_for_display[feature1] <= 300) &
                        (filtered_df_for_display[feature2] >= 18) & (filtered_df_for_display[feature2] <= 70)
                    ].copy()

                    color_map = {
                        'Low Risk (<50%)': 'green',
                        'Medium Risk (50-75%)': 'gold',
                        'High Risk (>75%)': 'red'
                    }

                    if not plot_df.empty:
                        # Use selected_risk_column_name for plotting color
                        fig_scatter = px.scatter(
                            plot_df,
                            x=feature1,
                            y=feature2,
                            color=selected_risk_column_name, # Use dynamic risk column
                            color_discrete_map=color_map,
                            hover_data=[feature1, feature2, selected_prob_column_for_plots],
                            title=f'Employee Attrition Risk Scatter Plot ({selected_prob_column_for_plots} by Hourly Comp vs Age)',
                            labels={feature1: feature1, feature2: feature2, selected_prob_column_for_plots: "Prob. Leave (%)", selected_risk_column_name: "Attrition Risk Level"} # Update label
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    else:
                        st.warning(f"No data to plot after filtering '{feature1}' between $5 and $300 and '{feature2}' between 18 and 70 with current risk filters.")
                else:
                    st.warning(f"Missing numerical columns for scatter plot: '{feature1}' or '{feature2}'")

                if 'State' in filtered_df_for_display.columns:
                    # Use selected_prob_column_for_plots for y-axis
                    state_risk = filtered_df_for_display.groupby('State')[selected_prob_column_for_plots].mean().reset_index()
                    state_risk = state_risk.sort_values(by=selected_prob_column_for_plots, ascending=False)

                    fig_bar = px.bar(
                        state_risk,
                        x='State',
                        y=selected_prob_column_for_plots,
                        color=selected_prob_column_for_plots,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title=f'Average {selected_prob_column_for_plots} by State',
                        labels={selected_prob_column_for_plots: "Prob. Leave (%)"},
                        hover_data={selected_prob_column_for_plots: ':.2f'}
                    )
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    # PASTE THE SNIPPET HERE (right after the above line)

                    # --- START Code for Top/Bottom N States Analysis ---
                    if not state_risk.empty:
                        N = 3 # You can make this configurable with st.number_input if desired
                        top_n_states = state_risk.head(N)
                        bottom_n_states = state_risk.tail(N).sort_values(by=selected_prob_column_for_plots, ascending=True)

                        st.markdown(f"**Top States with Highest Attrition Risk ({selected_timeframe}):**")
                        for idx, row in top_n_states.iterrows():
                            st.markdown(f"- **{row['State']}**: {row[selected_prob_column_for_plots]:.2f}%")

                        st.markdown(f"**Top States with Lowest Attrition Risk ({selected_timeframe}):**")
                        for idx, row in bottom_n_states.iterrows():
                            st.markdown(f"- **{row['State']}**: {row[selected_prob_column_for_plots]:.2f}%")

                        st.markdown(f"*(Based on average {selected_timeframe} consolidated probability among displayed employees.)*")
                    else:
                        st.info("Not enough state data for detailed top/bottom analysis based on current filters.")
                    # --- END Code for Top/Bottom N States Analysis ---
                else:
                    st.warning("Column 'State' not found in data for plotting.")

                if 'Home Department' in filtered_df_for_display.columns:
                    # Use selected_prob_column_for_plots for y-axis
                    dept_risk = filtered_df_for_display.groupby('Home Department')[selected_prob_column_for_plots].mean().reset_index()
                    dept_risk = dept_risk.sort_values(by=selected_prob_column_for_plots, ascending=False)

                    fig_dept_bar = px.bar(
                        dept_risk,
                        x='Home Department',
                        y=selected_prob_column_for_plots,
                        color=selected_prob_column_for_plots,
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title=f'Average {selected_prob_column_for_plots} by Home Department',
                        labels={selected_prob_column_for_plots: "Prob. Leave (%)"},
                        hover_data={selected_prob_column_for_plots: ':.2f'}
                    )
                    fig_dept_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_dept_bar, use_container_width=True)
                    # --- START Code for Department Attrition Textual Analysis ---
                    if not dept_risk.empty:
                        max_dept_prob = dept_risk[selected_prob_column_for_plots].max()
                        min_dept_prob = dept_risk[selected_prob_column_for_plots].min()
                        max_dept_name = dept_risk.loc[dept_risk[selected_prob_column_for_plots].idxmax(), 'Home Department']
                        min_dept_name = dept_risk.loc[dept_risk[selected_prob_column_for_plots].idxmin(), 'Home Department']

                        st.markdown(f"**Key Insights by Home Department:**")
                        st.markdown(f"- The highest average predicted attrition probability of **{max_dept_prob:.2f}%** is observed in **{max_dept_name}** department.")
                        st.markdown(f"- The lowest average predicted attrition probability of **{min_dept_prob:.2f}%** is observed in **{min_dept_name}** department.")
                        st.markdown(f"*(Based on {selected_timeframe} consolidated probability)*")
                    else:
                        st.info("No department data available for detailed analysis based on current filters.")
                    # --- END Code for Department Attrition Textual Analysis ---

                else:
                    st.warning("Column 'Home Department' not found in data for plotting.")
            else:
                st.info("No data available for visualizations based on current filters.")


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

        xgb_prob_leave = prediction_results.get("xgb_prob_leave")
        xgb_3mo = prediction_results.get("xgb_prob_leave_3mo")
        xgb_6mo = prediction_results.get("xgb_prob_leave_6mo")
        xgb_12mo = prediction_results.get("xgb_prob_leave_12mo")
        cox_3mo = prediction_results.get("cox_leave_3mo")
        cox_6mo = prediction_results.get("cox_leave_6mo")
        cox_12mo = prediction_results.get("cox_leave_12mo")

        if xgb_prob_leave is not None: # Check if at least the primary XGB prob is available
            # Calculate and Display Consolidated Probabilities
            st.write("**Consolidated Attrition Probabilities (Average of Models):**")
            consolidated_3mo = ((xgb_3mo if xgb_3mo is not None else 0) + (cox_3mo if cox_3mo is not None else 0)) / 2
            consolidated_6mo = ((xgb_6mo if xgb_6mo is not None else 0) + (cox_6mo if cox_6mo is not None else 0)) / 2
            consolidated_12mo = ((xgb_12mo if xgb_12mo is not None else 0) + (cox_12mo if cox_12mo is not None else 0)) / 2

            if xgb_3mo is not None or cox_3mo is not None:
                st.markdown(f"- **Prob. Leave (3 months):** **{consolidated_3mo * 100:.2f}%**")
            if xgb_6mo is not None or cox_6mo is not None:
                st.markdown(f"- **Prob. Leave (6 months):** **{consolidated_6mo * 100:.2f}%**")
            if xgb_12mo is not None or cox_12mo is not None:
                st.markdown(f"- **Prob. Leave (12 months):** **{consolidated_12mo * 100:.2f}%**")

            # Derive Attrition Risk Level for single prediction based on Consolidated 6mo Prob
            # This ensures consistency with batch predictions.
            temp_prob_for_risk = consolidated_6mo * 100
            attrition_risk_level_single = pd.cut(
                [temp_prob_for_risk],
                bins=[0, 49.99, 75, 100],
                labels=['Low Risk (<50%)', 'Medium Risk (50-75%)', 'High Risk (>75%)'],
                right=True,
                include_lowest=True
            )[0]
            st.markdown(f"- **Overall Attrition Risk Level (based on 6 months):** **{attrition_risk_level_single}**")

        else:
            st.error("Failed to get prediction for the single employee. Please check API connection and input data.")

        st.subheader("Input Data")
        st.json(single_row_data)

# --- START Code for Technical Model Explanation ---
st.markdown("---")  # Add a horizontal line for separation
st.header(" Why This Hybrid Attrition Model Is Technically Superior & Hard to Replicate")

st.markdown("""
### 1. **Dual-Model Power (Cox PH + XGBoost)**
Blends survival analysis with classification to predict **if** and **when** attrition may occur.  
While most models stop at binary outcomes (stay/leave), this model forecasts time-based probabilities — at **3, 6, and 12 months**.

### 2. **Advanced Survival Modeling**
The Cox Proportional Hazards model captures **ongoing employment** and **censoring effects**, yielding accurate risk even for active employees.

### 3. **Dynamic Tenure Simulation**
XGBoost predictions are **recalculated with simulated tenure changes**, enabling powerful “what-if” scenarios for proactive HR planning.

### 4. **Robust Preprocessing Engine**
Automatically **imputes missing values**, handles **rare categories**, and manages **unknown inputs**, making it resilient to messy real-world data.

### 5. **Time Normalization & Scaling**
Prediction windows are **normalized across tenure**, ensuring that risk is measured consistently regardless of employment length.

### 6. **Self-Healing & Modular Architecture**
Automatically **adds missing features** and **drops irrelevant ones** during inference, allowing upgrades without retraining the entire model.

### 7. **Enterprise-Grade Logging & Diagnostics**
Every key action — like imputation, mapping, or error handling — is **fully logged with context**, making production debugging straightforward.

### 8. **Versatile Input Handling**
Handles **nested JSON**, **stringified blobs**, and **batch requests** while maintaining **row-level alignment** — essential for integration into HR dashboards.

### 9. **Business-Ready Output**
Outputs are **human-readable and actionable**, like:  
 *3-month leave risk = 0.27*  
No data science team needed to interpret results.

### 10. **Deep Domain Intelligence**
Encodes **rehire logic**, **employee type-specific rules**, and **rare-state handling**, embedding years of expert knowledge that’s tough to replicate without proprietary data.
""")

# --- END Code for Technical Model Explanation ---

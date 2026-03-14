import streamlit as st
import os
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all preprocessing functions
from utils.preprocessing import (check_data_information, 
                                 initial_data_transform, 
                                 handle_missing_values, 
                                 filter_outliers, 
                                 feature_engineering, 
                                 feature_encoding, 
                                 feature_scaling)

from utils.feature_definitions import get_feature_definitions

#  Page Config
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title("Loan Prediction Analysis")

# Author Information
st.markdown("""
#### Author
Developed by : Muhammad Cikal Merdeka | Data Analyst/Data Scientist | Data Department

- [Email](mailto:mcikalmerdeka@gmail.com)
- [GitHub Profile](https://github.com/mcikalmerdeka)  
- [LinkedIn Profile](https://www.linkedin.com/in/mcikalmerdeka)
""")

# Add information about the app
with st.expander("**Read Instructions First: About This App**"):
    st.markdown("""
    ## Loan Default Prediction Application

    ### 📌 Purpose
    - This app uses machine learning to predict the creditworthiness of loan applicants.
    - The goal is to help financial institutions make more accurate lending decisions by identifying potential loan defaults.
    
    ### 🎯 Key Business Metrics
    Here are two critical metrics that could be improved through accurate loan prediction:

    #### Primary Metric: Default Rate (%)
    - Measures the percentage of customers who fail to repay their loans
    - Calculated as: (Number of Loan Defaults / Total Number of Customers) × 100
    - Lower default rate indicates more effective risk assessment
    - Critical for minimizing financial losses and improving lending strategies

    #### Secondary Metric: Approval Time
    - Tracks the time taken to process loan applications
    - Aims to streamline and accelerate the loan approval process
    - Reduces operational costs and improves customer satisfaction
    - Measures efficiency of the loan evaluation system

    ### 🔍 **How to Use the App**
                
    #### New Application Prediction:
    - Input new loan applicant details
    - Receive instant prediction of default probability

    #### Data Input Options:
    - A. Manual Input on Individual Customer Data
        - Enter details for a single loan applicant through the form
    - B. Upload Batch Data for Multiple Customers
        - Ensure your dataset matches the required structure for loan prediction (check the raw data preview)
        - Recommended columns/informations that you need to ensure exist include: income, assets information, profession, age, etc
    - Note: You can also use the example data provided for testing the model prediction capabilities

    #### Preprocessing Steps:
    The application will systematically process the loan application data through several crucial stages:
    1. Data Type Conversion
        - Standardize input data types from users for accurate analysis
    2. Missing Value Handling
        - Implement appropriate strategies for managing incomplete data
    3. Outlier Detection and Management
        - Identify and address extreme or anomalous data points
    4. Feature Engineering
        - Create derived features to enhance predictive power
        - Categorize continuous variables (e.g., age groups, income brackets)
    5. Feature Encoding
        - Convert categorical variables into numerical representations
    6. Feature Selection
        - Identify and retain most relevant predictors of loan default
    7. Data Scaling
        - Normalize features to ensure balanced model training
    All these steps are designed to optimize the model's predictive performance and ensure accurate loan default predictions.

    ### 🤖 Model Capabilities Information (Additional Info For Developers)
    - This app uses a tuned K-Nearest Neighbors (KNN) model to predict loan default probability
    - The model have recall score of 97.97 ± 0.06 (training) and 85.88 ± 0.23 (testing)
    - The model is trained on a dataset of 32k loan applications
    - More info on the model training process can be found in the project repository

    ### ⚠️ <span style="color:red;"> Important Notes </span>
    - Model predictions are probabilistic and should be used as a decision support tool
    - **Final lending decisions should combine model insights with human expertise**
    - Continuous monitoring and updating of the model is recommended to maintain performance
    """, unsafe_allow_html=True)

# Cache expensive data loading operations
@st.cache_data
def load_column_metadata():
    """Load and cache column metadata from training data to avoid loading full dataset repeatedly."""
    df = pd.read_csv('data/Training Data.csv')
    df = initial_data_transform(df)
    
    # Extract only the metadata we need for the form
    metadata = {}
    for col in df.columns:
        if col != "Risk_Flag":
            metadata[col] = {
                'dtype': str(df[col].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(df[col]),
                'is_categorical': pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object',
                'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'unique_values': df[col].unique().tolist() if not pd.api.types.is_numeric_dtype(df[col]) else None
            }
    
    # Also get sample data for preview
    sample_data = df.head()
    
    return metadata, sample_data

@st.cache_data
def load_preprocessed_columns():
    """Load and cache the expected columns from preprocessed data."""
    df = pd.read_csv('data/df_model_rewrite.csv')
    columns = [col for col in df.columns if col != "Risk_Flag"]
    return columns

# Load pre-trained model
@st.cache_resource
def load_model():
    # Model is in the models directory at project root
    model_path = os.path.join('models', 'tuned_knn_model.joblib')
    return joblib.load(model_path)

model = load_model()

# Load cached metadata instead of full datasets
column_metadata, ori_df_sample = load_column_metadata()
expected_columns = load_preprocessed_columns()

# Display original data
st.subheader("Original Data Preview")
st.write(ori_df_sample)

# Display data information
with st.expander("📊 Data Information"):
    st.markdown("### Data Information")
    # Convert metadata to DataFrame for display
    info_data = []
    for col, meta in column_metadata.items():
        info_data.append({
            'Feature': col,
            'Data Type': meta['dtype'],
            'Unique Values': str(len(meta['unique_values'])) if meta['unique_values'] else 'N/A',
            'Sample Values': str(meta['unique_values'][:5]) if meta['unique_values'] else 'N/A'
        })
    st.write(pd.DataFrame(info_data))

# Add Data Dictionary section
with st.expander("📚 Data Dictionary"):
    st.markdown("### Feature Information")
    
    # Create DataFrame from feature definitions
    definitions = get_feature_definitions()
    feature_df = pd.DataFrame.from_dict(definitions, orient='index')
    
    # Reorder columns and reset index to show feature names as a column
    feature_df = feature_df.reset_index().rename(columns={'index': 'Feature Name'})
    feature_df = feature_df[['Feature Name', 'description', 'data_type', 'specific_type']]
    
    # Rename columns for display
    feature_df.columns = ['Feature Name', 'Description', 'Data Type', 'Specific Type']
    
    # Display as a styled table
    st.dataframe(
        feature_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'lightgrey'
        })
    )
    
    st.markdown("""
    **Note:**
    - Categorical (Nominal): Categories without any natural order
    - Categorical (Ordinal): Categories with a natural order
    - Numerical (Discrete): Whole numbers
    - Numerical (Continuous): Any numerical value
    """)

## Specifying some variable values for the model and code flow
target_col = "Risk_Flag"
gather_data = False

# Input type selection
input_type = st.empty()
input_type = st.radio('Select Input Type', ['Individual Customer', 'Batch Data'])
if input_type.lower() == 'individual customer':
    st.write('Please provide the details of the customer in the form below')

    # Input individual customer data
    st.subheader("Enter Customer Data")
    with st.form("customer_prediction_form"):
        # Create a dictionary to store input values
        prediction_input = {}

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        # Split columns into two groups for layout
        all_columns = list(column_metadata.keys())
        mid_point = len(all_columns) // 2

        with col1:
            for column in all_columns[:mid_point]:
                meta = column_metadata[column]
                
                if meta['is_datetime']:
                    prediction_input[column] = st.date_input(f"Enter {column}")

                elif meta['is_numeric']:
                    col_min = meta['min'] if meta['min'] is not None else 0.0
                    col_max = meta['max'] if meta['max'] is not None else None
                    col_mean = meta['mean'] if meta['mean'] is not None else 0.0

                    prediction_input[column] = st.number_input(
                        f"Enter {column}",
                        min_value=float(col_min),
                        max_value=float(col_max) if col_max is not None else None,
                        value=float(col_mean),
                        step=0.1
                    )
                    
                elif meta['is_categorical']:
                    unique_values = meta['unique_values']
                    prediction_input[column] = st.selectbox(
                        f'Select {column}',
                        options=unique_values
                    )
                
                else:
                    prediction_input[column] = st.text_input(f'Enter {column}')

        with col2:
            for column in all_columns[mid_point:]:
                meta = column_metadata[column]
                
                if meta['is_datetime']:
                    prediction_input[column] = st.date_input(f"Enter {column}")
                
                elif meta['is_numeric']:
                    col_min = meta['min'] if meta['min'] is not None else 0.0
                    col_max = meta['max'] if meta['max'] is not None else None
                    col_mean = meta['mean'] if meta['mean'] is not None else 0.0

                    prediction_input[column] = st.number_input(
                        f"Enter {column}",
                        min_value=float(col_min),
                        max_value=float(col_max) if col_max is not None else None,
                        value=float(col_mean),
                        step=0.1
                    )
                    
                elif meta['is_categorical']:
                    unique_values = meta['unique_values']
                    prediction_input[column] = st.selectbox(
                        f'Select {column}',
                        options=unique_values
                    )
                
                else:
                    prediction_input[column] = st.text_input(f'Enter {column}')
            
        # Add hint for user testing
        with st.expander("📌 Hint for Testing Model Prediction"):
            st.write("You can use the example data below as a reference for input values:")
            
            # Example data of a customer who is predicted as Not Default
            example_data_1 = {
                "id": 1,
                "Income": 8877323,
                "Age": 60,
                "Experience": 12,
                "Marital_Status": "single",
                "House_Ownership": "owned",
                "Car_Ownership": "Yes",
                "Profession": "software developer",
                "City": "rewa",
                "State": "tamil nadu",
                "Current_Job_Years": 12,
                "Current_House_Years": 12,
            }
            st.table(pd.DataFrame([example_data_1]))
            st.write("Which will result in a prediction of <span style='color:green;'>**Not Default**</span>", unsafe_allow_html=True)
            
            # Example data of a customer who is predicted as Default
            example_data_2 = {
                "id": 2,
                "Income": 3234134,
                "Age": 25,
                "Experience": 2,
                "Marital_Status": "single",
                "House_Ownership": "rented",
                "Car_Ownership": "no",
                "Profession": "police officer",
                "City": "rewa",
                "State": "madhya pradesh",
                "Current_Job_Years": 2,
                "Current_House_Years": 10,
            }
            st.table(pd.DataFrame([example_data_2]))
            st.write("Which will result in a prediction of <span style='color:red;'>**Default**</span>", unsafe_allow_html=True)

            st.write("Note: You can see the behaviour of the model and how it prefer certain values to be predicted as default or not default")

        # Submit button
        submit_prediction_button = st.form_submit_button("Predict Customer Loan Status")
        gather_data = True

elif input_type.lower() == 'batch data':
    st.write('Please upload the dataset of the customers \n\n Ensure your dataset matches the required structure for loan prediction (check the example data preview, exclude the target column)')

    # File upload
    uploaded_data = st.file_uploader("Choose a CSV file (**Please make sure you convert it to csv first**)", type="csv")

    if uploaded_data is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_data)
            batch_input_df = initial_data_transform(batch_input_df)
            st.success("File uploaded successfully")
            gather_data = True

        except Exception as e:
            st.error(f"Error uploading the file: {str(e)}")

    # Add hint for user testing
    with st.expander("📌 Hint for Testing Model Prediction"):
        st.write("You can use the example data by clicking this button below as a reference for input values:")

        # First button with a unique key
        if st.button("Use Example Data", key="example_data_button"):
            # Load example CSV data from local data directory
            batch_input_df = pd.read_csv('data/batch_example.csv')
            batch_input_df = initial_data_transform(batch_input_df)
            gather_data = True
        
# Prediction Section

## Prediction for individual customer
if gather_data and input_type.lower() == 'individual customer':
    if submit_prediction_button:
        # Convert input data into dataframe
        input_df = pd.DataFrame([prediction_input])

        # Show input data
        st.subheader("New Customer Input Data Preview")
        st.write(input_df)

        # Preprocessing steps

        ## 1. Handle Missing Values
        try:
            input_df = handle_missing_values(input_df, columns=None, strategy='fill', imputation_method='median')
        except Exception as e:
            st.error(f"Error in handling missing values: {str(e)}")

        ## 2. Handle Outliers
        try:
            input_df = filter_outliers(input_df, col_series=None, method='iqr')
        except Exception as e:
            st.error(f"Error in handling outliers: {str(e)}")

        ## 3. Feature Engineering
        try:
            input_df = feature_engineering(input_df)
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")

        # Check data after feature engineering
        st.subheader("After Feature Engineering")
        st.write(input_df) 

        ## 4. Feature encoding
        try:
            input_df = feature_encoding(input_df, expected_columns=expected_columns)
        except Exception as e:
            st.error(f"Error in feature encoding: {str(e)}")
            st.write("Debug information:")
            st.write("Current columns:", input_df.columns.to_list())
            st.write("Expected columns:", expected_columns)

        # Check data after encoding
        st.subheader("After Feature Encoding and Drop Columns")
        st.write(input_df)
        
        ## 5. Feature Scaling
        try:
            input_df = feature_scaling(data=input_df)
        except Exception as e:
            st.error(f"Error in feature scaling: {str(e)}")

        # Check data after scaling
        st.subheader("After Feature Scaling")
        st.write(input_df)

        # Prediction Section
        st.subheader("Prediction Section")

        # Create a copy for preprocessing result
        model_df = input_df.copy()

        # Display the prediction result
        try:
            prediction = model.predict(model_df)

            # Display prediction result with explanation
            if prediction[0] == 0:
                st.success("The customer is predicted as **Not Default**.\n\n**Not Default** means the customer is likely to repay the loan on time.")
            else:
                st.error("The customer is predicted as **Default**.\n\n**Default** means the customer is likely to fail to repay the loan on time.")
        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

## Prediction for batch data
elif gather_data and input_type.lower() == 'batch data':
        # Show batch input data
        st.subheader("Batch Data Preview")
        st.write(batch_input_df.head())

        # Preprocessing steps

        ## 1. Handle Missing Values
        try:
            batch_input_df = handle_missing_values(batch_input_df, columns=None, strategy='fill', imputation_method='median')
        except Exception as e:
            st.error(f"Error in handling missing values: {str(e)}")

        ## 2. Handle Outliers
        try:
            batch_input_df = filter_outliers(batch_input_df, col_series=None, method='iqr')
        except Exception as e:
            st.error(f"Error in handling outliers: {str(e)}")

        ## 3. Feature Engineering
        try:
            batch_input_df = feature_engineering(batch_input_df)
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")

        # Check data after feature engineering
        st.subheader("After Feature Engineering")
        st.write(batch_input_df) 

        ## 4. Feature encoding
        try:
            batch_input_df = feature_encoding(batch_input_df, expected_columns=expected_columns)
        except Exception as e:
            st.error(f"Error in feature encoding: {str(e)}")
            st.write("Debug information:")
            st.write("Current columns:", batch_input_df.columns.to_list())
            st.write("Expected columns:", expected_columns)

        # Check data after encoding
        st.subheader("After Feature Encoding and Drop Columns")
        st.write(batch_input_df)
        
        ## 5. Feature Scaling
        try:
            batch_input_df = feature_scaling(data=batch_input_df)
        except Exception as e:
            st.error(f"Error in feature scaling: {str(e)}")

        # Check data after scaling
        st.subheader("After Feature Scaling")
        st.write(batch_input_df)        

        # Prediction Section
        st.subheader("Prediction Section")

        # Create a copy for preprocessing result
        model_df = batch_input_df.copy()

        # Display the prediction result - predict all rows at once for better performance
        try:
            predictions = model.predict(model_df)
            
            # Create a results dataframe
            results_df = pd.DataFrame({
                'Customer_ID': range(1, len(predictions) + 1),
                'Prediction': ['Not Default' if p == 0 else 'Default' for p in predictions]
            })
            
            # Display summary
            st.write(f"**Total Predictions: {len(predictions)}**")
            st.write(f"- Not Default: {sum(predictions == 0)}")
            st.write(f"- Default: {sum(predictions == 1)}")
            
            # Display individual results in an expander to avoid cluttering the UI
            with st.expander("View Individual Predictions"):
                for idx, (row_num, pred) in enumerate(zip(range(1, len(predictions) + 1), predictions)):
                    if pred == 0:
                        st.success(f"Customer {row_num}: **Not Default**")
                    else:
                        st.error(f"Customer {row_num}: **Default**")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
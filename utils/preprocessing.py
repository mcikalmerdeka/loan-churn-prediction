# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

# Import the preprocessed original data
url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/main/data/df_model_rewrite.csv"
ori_df_preprocessed = pd.read_csv(url_ori_processed)
ori_df_preprocessed = ori_df_preprocessed.loc[:, ori_df_preprocessed.columns != "Risk_Flag"]

# =====================================================================Functions for data pre-processing========================================================================

## Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df

## Initial data transformation
def initial_data_transform(data):
    # Rename some columns
    data = data.rename(columns={'CURRENT_JOB_YRS' : 'Current_Job_Years',
                                'CURRENT_HOUSE_YRS' : 'Current_House_Years',
                                'CITY' : 'City',
                                'STATE' : 'State',
                                'Married/Single' : 'Marital_Status'})
    
    # Clean invalid characters and lowercase the values of categorical columns
    columns_to_clean = ['Profession', 'State', 'City']

    # Removing those characters from the 'Profession', 'City', and 'State' column
    for col in columns_to_clean:
        data[col] = data[col].str.replace(r'\[\d+\]', '', regex=True)
        data[col] = data[col].str.replace('_', ' ')
        data[col] = data[col].str.replace(',', ' ')

    # Rename the format of the values in those columns to title
    for col in columns_to_clean:
        data[col] = data[col].str.lower()

    # Replace Delhi_city city to New_Delhi
    data['City'] = data['City'].replace('delhi city', 'new delhi')

    return data

## Impute missing values function
def handle_missing_values(data, columns, strategy='fill', imputation_method='median'):
    # Return the original data if the column is empty
    if columns is None:
        return data
    
    # Impute missing values based on the strategy
    if strategy == 'fill':
        if imputation_method == 'median':
            return data[columns].fillna(data[columns].median())
        elif imputation_method == 'mean':
            return data[columns].fillna(data[columns].mean())
        elif imputation_method == 'mode':
            return data[columns].fillna(data[columns].mode().iloc[0])
        elif imputation_method == 'ffill':
            return data[columns].fillna(method='ffill')
        elif imputation_method == 'bfill':
            return data[columns].fillna(method='bfill')
        else:
            return data[columns].fillna(data[columns].median())

    # Remove rows with missing values
    elif strategy == 'remove':
        return data.dropna(subset=columns)
    
## Drop columns function
def drop_columns(data, columns):
    return data.drop(columns=columns, errors='ignore')

## Handle outliers function
def filter_outliers(data, col_series, method='iqr', threshold=3):
    # Return the original data if the column series is empty
    if col_series is None:
        return data

    # Validate the method parameter
    if method.lower() not in ['iqr', 'zscore']:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    # Start with all rows marked as True (non-outliers)
    filtered_entries = np.array([True] * len(data))
    
    # Loop through each column
    for col in col_series:
        if method.lower() == 'iqr':
            # IQR method
            Q1 = data[col].quantile(0.25)  # First quartile (25th percentile)
            Q3 = data[col].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - (IQR * 1.5)  # Lower bound for outliers
            upper_bound = Q3 + (IQR * 1.5)  # Upper bound for outliers

            # Create a filter that identifies non-outliers for the current column
            filter_outlier = ((data[col] >= lower_bound) & (data[col] <= upper_bound))
            
        elif method.lower() == 'zscore':  # zscore method
            # Calculate Z-Scores and create filter
            z_scores = np.abs(stats.zscore(data[col]))

            # Create a filter that identifies non-outliers
            filter_outlier = (z_scores < threshold)
        
        # Update the filter to exclude rows that have outliers in the current column
        filtered_entries = filtered_entries & filter_outlier
    
    return data[filtered_entries]
    
## Feature engineering function (Create only used feature in the model not all from the notebook)
def feature_engineering(data):
        # A. Generation
        def assign_generation(age):
            if age <= 27:
                return 'Generation Z'
            elif age <= 43:
                return 'Generation Millenials'
            elif age < 59:
                return 'Generation X'
            elif age < 69:
                return 'Boomers II'
            elif age <= 78:
                return 'Boomers I'
            else:
                return 'Other'

        data['Generation'] = data['Age'].apply(assign_generation)

        # Ratio Experience by Age
        data['Experience_Age_Ratio'] = data['Experience'] / data['Age']

        # B. Profession grouping
        profession_groups = {
        'engineering': ['engineer', 'mechanical engineer', 'civil engineer', 'industrial engineer', 'design engineer', 'chemical engineer', 'biomedical engineer', 'computer hardware engineer', 'petroleum engineer', 'surveyor', 'drafter'],
        'technology': ['software developer', 'computer operator', 'technology specialist', 'web designer', 'technician'],
        'healthcare': ['physician', 'dentist', 'surgeon', 'psychologist'],
        'finance': ['economist', 'financial analyst', 'chartered accountant'],
        'design': ['architect', 'designer', 'graphic designer', 'fashion designer', 'artist'],
        'aviation': ['flight attendant', 'air traffic controller', 'aviator'],
        'government public service': ['civil servant', 'politician', 'police officer', 'magistrate', 'army officer', 'firefighter', 'lawyer', 'official', 'librarian'],
        'business management' : ['hotel manager', 'consultant', 'secretary'],
        'science research' : ['scientist', 'microbiologist', 'geologist', 'statistician', 'analyst'],
        'miscellaneous': ['comedian', 'chef', 'technical writer']}

        data['Profession_Group'] = data['Profession'].map({prof: group for group, prof_list in profession_groups.items() for prof in prof_list})

        # C. State grouping
        def state_group(state) :
            if state in ['uttar pradesh', 'haryana', 'jammu and kashmir', 'punjab', 'uttarakhand', 'chandigarh', 'delhi', 'himachal pradesh'] :
                return 'north_zone'
            elif state in ['bihar', 'jharkhand', 'odisha', 'west bengal', 'assam', 'sikkim', 'tripura', 'mizoram', 'manipur'] :
                return 'east_zone'
            elif state in ['andhra pradesh', 'tamil nadu', 'karnataka', 'telangana', 'kerala', 'puducherry'] :
                return 'south_zone'
            else :
                return 'west_zone'

        data['State_Group'] = data['State'].apply(state_group)

        # D. City grouping
        def city_group(city):
            if city in ['new delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore']:
                return 'metro'
            elif city in ['ahmedabad', 'hyderabad', 'pune', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'visakhapatnam', 'indore', 'thane',
                        'bhopal', 'pimpri-chinchwad', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot',
                        'varanasi', 'srinagar', 'amritsar', 'allahabad', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'raipur', 'kota', 'guwahati', 'chandigarh city']:
                return 'urban'
            elif city in ['navi mumbai', 'kalyan-dombivli', 'vasai-virar', 'mira-bhayandar', 'thiruvananthapuram', 'bhiwandi', 'noida', 'bhopal', 'howrah', 'saharanpur',
                        'berhampur', 'suryapet', 'muzaffarpur', 'nadiad', 'siliguri', 'bhavnagar', 'kurnool', 'tenali', 'satna', 'nandyal', 'etawah', 'morena', 'ballia',
                        'machilipatnam', 'mau', 'machilipatnam', 'bhagalpur', 'siwan', 'meerut', 'dibrugarh', 'gaya', 'darbhanga', 'hajipur', 'mirzapur', 'akola', 'satna',
                        'motihari', 'jalna', 'ramgarh', 'ozhukarai', 'saharsa', 'munger', 'farrukhabad', 'nangloi jat', 'thoothukudi', 'nagercoil', 'rourkela', 'jhansi', 'sultan pur majra']:
                return 'suburban'
            else:
                return 'rural'

        data['City_Group'] = data['City'].apply(city_group)

        return data
    
## Feature encoding function
def feature_encoding(data, original_data=ori_df_preprocessed):
        # A. Handle ordinal encoding for Car_Ownership and Generation (unchanged)
        data['Car_Ownership'] = data['Car_Ownership'].map({'no': 0, 'yes': 1})

        data['Generation'] = data['Generation'].map({'Generation Z': 0,
                                                            'Generation Millenials': 1,
                                                            'Generation X' : 2,
                                                            'Boomers II' : 3,
                                                            'Boomers I' : 4,
                                                            'Other' : 5})
        
        # B. Handle one-hot encoding for Profession using original data categories
        unique_professions = original_data.filter(like='Prof_').columns
        prof_encoded = pd.DataFrame(0, index=data.index, columns=unique_professions)
        if f"Prof_{data['Profession_Group'].iloc[0]}" in unique_professions:
            prof_encoded[f"Prof_{data['Profession_Group'].iloc[0]}"] = 1
        data = drop_columns(data, ['Profession_Group'])
        data = pd.concat([data, prof_encoded], axis=1)

        # C. Handle one-hot encoding for State using original data categories
        unique_states = original_data.filter(like='State_').columns
        state_encoded = pd.DataFrame(0, index=data.index, columns=unique_states)
        if f"State_{data['State_Group'].iloc[0]}" in unique_states:
            state_encoded[f"State_{data['State_Group'].iloc[0]}"] = 1
        data = drop_columns(data, ['State_Group'])
        data = pd.concat([data, state_encoded], axis=1)

        # D. Handle one-hot encoding for City using original data categories
        unique_cities = original_data.filter(like='City_').columns
        city_encoded = pd.DataFrame(0, index=data.index, columns=unique_cities)
        if f"City_{data['City_Group'].iloc[0]}" in unique_cities:
            city_encoded[f"City_{data['City_Group'].iloc[0]}"] = 1
        data = drop_columns(data, ['City_Group'])
        data = pd.concat([data, city_encoded], axis=1)

        # Ensure all expected columns are present before moving to scaling
        expected_columns = original_data.columns.tolist()
        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0

        for col in data.columns:
            if col not in expected_columns:
                data.drop(columns=col, inplace=True)

        # Reorder and match columns to match training data
        data = data[expected_columns]

        return data, expected_columns

## Feature scaling function
def feature_scaling(data, original_data=ori_df_preprocessed):
        
        # Initialize scalers
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        # Define feature groups for targeted scaling
        # Each feature group requires a specific scaling approach
        log_transform_features = ['Experience_Age_Ratio']  # Features with high skewness
        count_uniform_features = ['Income']  # Discrete features representing quantities

        scalers = {}  # Dictionary to store fitted scalers and feature info

        # TRAINING DATA SCALING
        # Step 1: Scale count/uniform features using MinMaxScaler
        original_data[count_uniform_features] = minmax_scaler.fit_transform(original_data[count_uniform_features])
        scalers['count_uniform'] = minmax_scaler

        # Step 2: Scale skewed features using log transformation and standardization

        original_data[log_transform_features] = np.log1p(original_data[log_transform_features])
        original_data[log_transform_features] = standard_scaler.fit_transform(original_data[log_transform_features])
        scalers['log_transform'] = standard_scaler

        # INFERENCE DATA SCALING
        # Apply the same transformations used in training data
        # Use .transform() instead of .fit_transform() to maintain training distribution

        # Scale count/uniform features
        data[count_uniform_features] = scalers['count_uniform'].transform(data[count_uniform_features])

        # Scale skewed features
        data[log_transform_features] = np.log1p(data[log_transform_features])
        data[log_transform_features] = scalers['log_transform'].transform(data[log_transform_features])

        return data
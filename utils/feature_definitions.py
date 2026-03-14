def get_feature_definitions():
    return {
        "Id": {
            "description": "Unique identifier of the user",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Income": {
            "description": "Annual income of the user",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Age": {
            "description": "Age of the user",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Experience": {
            "description": "Total professional experience in years",
            "data_type": "Numerical",
            "specific_type": "Continuous"
        },
        "Married/Single": {
            "description": "Marital status of the user",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "House_Ownership": {
            "description": "Status of housing ownership",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Car_Ownership": {
            "description": "Vehicle ownership status",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "Profession": {
            "description": "User's professional occupation",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "CITY": {
            "description": "City of residence",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "STATE": {
            "description": "State of residence",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        },
        "CURRENT_JOB_YRS": {
            "description": "Years of experience in current job",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "CURRENT_HOUSE_YRS": {
            "description": "Years in current residence",
            "data_type": "Numerical",
            "specific_type": "Discrete"
        },
        "Risk_Flag": {
            "description": "Loan default status (0: Not Defaulted, 1: Defaulted)",
            "data_type": "Categorical",
            "specific_type": "Nominal"
        }
    }
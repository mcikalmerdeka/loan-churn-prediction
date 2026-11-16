"""Dataset schema and feature metadata for the loan prediction app.

This module is the single source of truth for column names, model inputs,
grouping rules, and the data dictionary. It has no Streamlit or modelling
dependencies, so the notebook, pipeline, and views can all import it without
creating a cycle.
"""

from __future__ import annotations

from typing import Dict, List

TARGET_COLUMN = "Risk_Flag"
ID_COLUMN = "Id"

OUTCOME_LABELS = {0: "Not Default", 1: "Default"}

# Raw fields the deployed model actually consumes. Everything else in the
# source dataset is either unused or only used to derive a model feature.
REQUIRED_INPUT_COLUMNS = [
    "Income",
    "Age",
    "Experience",
    "Car_Ownership",
    "Profession",
    "City",
    "State",
    "Current_House_Years",
]

UNUSED_INPUT_COLUMNS = [
    "Marital_Status",
    "House_Ownership",
    "Current_Job_Years",
]

# Order matters: this is the exact matrix the persisted KNN was fitted on.
MODEL_FEATURES = [
    "Income",
    "Car_Ownership",
    "Current_House_Years",
    "Generation",
    "Prof_business management",
    "Prof_design",
    "Prof_engineering",
    "Prof_finance",
    "Prof_government public service",
    "Prof_healthcare",
    "Prof_miscellaneous",
    "Prof_science research",
    "Prof_technology",
    "State_north_zone",
    "State_south_zone",
    "State_west_zone",
    "City_rural",
    "City_suburban",
    "City_urban",
    "Experience_Age_Ratio",
]

# The notebook's column renames, plus the typo present in batch_example.csv.
COLUMN_RENAMES = {
    "CURRENT_JOB_YRS": "Current_Job_Years",
    "CURRENT_HOUSE_YRS": "Current_House_Years",
    "CITY": "City",
    "STATE": "State",
    "Married/Single": "Marital_Status",
    "Marita_Status": "Marital_Status",
}

PROFESSION_GROUPS: Dict[str, List[str]] = {
    "engineering": [
        "engineer",
        "mechanical engineer",
        "civil engineer",
        "industrial engineer",
        "design engineer",
        "chemical engineer",
        "biomedical engineer",
        "computer hardware engineer",
        "petroleum engineer",
        "surveyor",
        "drafter",
    ],
    "technology": [
        "software developer",
        "computer operator",
        "technology specialist",
        "web designer",
        "technician",
    ],
    "healthcare": ["physician", "dentist", "surgeon", "psychologist"],
    "finance": ["economist", "financial analyst", "chartered accountant"],
    "design": ["architect", "designer", "graphic designer", "fashion designer", "artist"],
    "aviation": ["flight attendant", "air traffic controller", "aviator"],
    "government public service": [
        "civil servant",
        "politician",
        "police officer",
        "magistrate",
        "army officer",
        "firefighter",
        "lawyer",
        "official",
        "librarian",
    ],
    "business management": ["hotel manager", "consultant", "secretary"],
    "science research": [
        "scientist",
        "microbiologist",
        "geologist",
        "statistician",
        "analyst",
    ],
    "miscellaneous": ["comedian", "chef", "technical writer"],
}

PROFESSION_TO_GROUP = {
    profession: group
    for group, professions in PROFESSION_GROUPS.items()
    for profession in professions
}

STATE_ZONES: Dict[str, List[str]] = {
    "north_zone": [
        "uttar pradesh",
        "haryana",
        "jammu and kashmir",
        "punjab",
        "uttarakhand",
        "chandigarh",
        "delhi",
        "himachal pradesh",
    ],
    "east_zone": [
        "bihar",
        "jharkhand",
        "odisha",
        "west bengal",
        "assam",
        "sikkim",
        "tripura",
        "mizoram",
        "manipur",
    ],
    "south_zone": [
        "andhra pradesh",
        "tamil nadu",
        "karnataka",
        "telangana",
        "kerala",
        "puducherry",
    ],
    "west_zone": [],
}

STATE_ZONE_MAP = {
    state: zone for zone, states in STATE_ZONES.items() for state in states
}
DEFAULT_STATE_ZONE = "west_zone"

CITY_GROUPS: Dict[str, List[str]] = {
    "metro": ["new delhi", "mumbai", "kolkata", "chennai", "bangalore"],
    "urban": [
        "ahmedabad",
        "hyderabad",
        "pune",
        "surat",
        "jaipur",
        "lucknow",
        "kanpur",
        "nagpur",
        "visakhapatnam",
        "indore",
        "thane",
        "bhopal",
        "pimpri-chinchwad",
        "patna",
        "vadodara",
        "ghaziabad",
        "ludhiana",
        "agra",
        "nashik",
        "faridabad",
        "meerut",
        "rajkot",
        "varanasi",
        "srinagar",
        "amritsar",
        "allahabad",
        "jabalpur",
        "gwalior",
        "vijayawada",
        "jodhpur",
        "raipur",
        "kota",
        "guwahati",
        "chandigarh city",
    ],
    "suburban": [
        "navi mumbai",
        "kalyan-dombivli",
        "vasai-virar",
        "mira-bhayandar",
        "thiruvananthapuram",
        "bhiwandi",
        "noida",
        "howrah",
        "saharanpur",
        "berhampur",
        "suryapet",
        "muzaffarpur",
        "nadiad",
        "siliguri",
        "bhavnagar",
        "kurnool",
        "tenali",
        "satna",
        "nandyal",
        "etawah",
        "morena",
        "ballia",
        "machilipatnam",
        "mau",
        "bhagalpur",
        "siwan",
        "dibrugarh",
        "gaya",
        "darbhanga",
        "hajipur",
        "mirzapur",
        "akola",
        "motihari",
        "jalna",
        "ramgarh",
        "ozhukarai",
        "saharsa",
        "munger",
        "farrukhabad",
        "nangloi jat",
        "thoothukudi",
        "nagercoil",
        "rourkela",
        "jhansi",
        "sultan pur majra",
    ],
    "rural": [],
}
DEFAULT_CITY_GROUP = "rural"

# Build the lookup with the notebook's priority: metro, then urban, then
# suburban. A few names appear in both urban and suburban lists (bhopal,
# meerut, satna, machilipatnam); the notebook checks urban first.
CITY_GROUP_MAP: Dict[str, str] = {}
for _priority_group in ("metro", "urban", "suburban"):
    for _city in CITY_GROUPS[_priority_group]:
        CITY_GROUP_MAP.setdefault(_city, _priority_group)

GENERATION_CATEGORIES = [
    "Generation Z",
    "Generation Millenials",
    "Generation X",
    "Boomers II",
    "Boomers I",
    "Other",
]

CAR_OWNERSHIP_VALUES = ["no", "yes"]

NUMERIC_PROFILE_COLUMNS = [
    "Income",
    "Age",
    "Experience",
    "Current_House_Years",
]

USED_BY_MODEL = set(REQUIRED_INPUT_COLUMNS)
DERIVED_BY_MODEL = {
    "Generation",
    "Experience_Age_Ratio",
    "Profession_Group",
    "State_Group",
    "City_Group",
}


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """Data dictionary rendered by the app's About tab."""
    return {
        "Id": {
            "description": "Unique identifier of the applicant",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Income": {
            "description": "Annual income of the applicant",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Age": {
            "description": "Age of the applicant",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Experience": {
            "description": "Total professional experience in years",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Marital_Status": {
            "description": "Marital status of the applicant",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "House_Ownership": {
            "description": "Housing ownership status",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Car_Ownership": {
            "description": "Vehicle ownership status",
            "data_type": "Categorical",
            "specific_type": "Binary",
        },
        "Profession": {
            "description": "Applicant's occupation",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "City": {
            "description": "City of residence",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "State": {
            "description": "State of residence",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Current_Job_Years": {
            "description": "Years in the current job",
            "data_type": "Numerical",
            "specific_type": "Discrete",
        },
        "Current_House_Years": {
            "description": "Years in the current residence",
            "data_type": "Numerical",
            "specific_type": "Discrete",
        },
        "Generation": {
            "description": "Generation group derived from Age",
            "data_type": "Categorical",
            "specific_type": "Ordinal (derived)",
        },
        "Experience_Age_Ratio": {
            "description": "Experience divided by Age",
            "data_type": "Numerical",
            "specific_type": "Continuous (derived)",
        },
        "Profession_Group": {
            "description": "Occupation grouped into ten families",
            "data_type": "Categorical",
            "specific_type": "Nominal (derived)",
        },
        "State_Group": {
            "description": "State grouped into north / east / south / west zones",
            "data_type": "Categorical",
            "specific_type": "Nominal (derived)",
        },
        "City_Group": {
            "description": "City grouped into metro / urban / suburban / rural",
            "data_type": "Categorical",
            "specific_type": "Ordinal (derived)",
        },
        "Risk_Flag": {
            "description": "Loan default status (1 = Default, 0 = Not Default)",
            "data_type": "Categorical",
            "specific_type": "Nominal (binary target)",
        },
    }

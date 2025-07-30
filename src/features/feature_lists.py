"""
Mixed manually/automatically generated feature lists (Day 9 – Step 2)
------------------------------------------------
numeric_features      : Continuous/discrete numeric columns
categorical_low_card  : Cardinality ≤ 15
categorical_high_card : Cardinality > 15
"""

numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

categorical_low_card = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

categorical_high_card = []

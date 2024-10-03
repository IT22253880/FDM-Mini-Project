
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib  # Import joblib if you want to load a pre-trained model

# Define the user input
st.title('Loan Default Prediction')

# Add input fields for each feature
score_source_3 = st.number_input('Score Source 3', min_value=0.0)
active_loan = st.number_input('Active Loan', min_value=0)
score_source_1 = st.number_input('Score Source 1', min_value=0.0)
bike_owned = st.number_input('Bike Owned', min_value=0)
house_owned = st.number_input('House Owned', min_value=0)
client_city_rating = st.number_input('Client City Rating', min_value=0)
score_source_2 = st.number_input('Score Source 2', min_value=0.0)
credit_bureau = st.number_input('Credit Bureau', min_value=0)
application_process_day = st.number_input('Application Process Day', min_value=0)
credit_amount = st.number_input('Credit Amount', min_value=0.0)
population_region_relative = st.number_input('Population Region Relative', min_value=0)
own_house_age = st.number_input('Own House Age', min_value=0)
child_count = st.number_input('Child Count', min_value=0)
registration_days = st.number_input('Registration Days', min_value=0)
employed_days = st.number_input('Employed Days', min_value=0)

# Prepare the input data for prediction
input_data = pd.DataFrame([[score_source_3, active_loan, score_source_1, bike_owned, house_owned,
                            client_city_rating, score_source_2, credit_bureau,
                            application_process_day, credit_amount,
                            population_region_relative, own_house_age,
                            child_count, registration_days, employed_days]],
                           columns=['Score_Source_3', 'Active_Loan', 'Score_Source_1', 'Bike_Owned',
                                    'House_Own', 'Client_City_Rating', 'Score_Source_2',
                                    'Credit_Bureau', 'Application_Process_Day', 'Credit_Amount',
                                    'Population_Region_Relative', 'Own_House_Age', 'Child_Count',
                                    'Registration_Days', 'Employed_Days'])

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Load your trained model here if available
# rf_model = joblib.load('your_model_file.pkl')

# Create a button to trigger prediction
if st.button('Submit and Predict'):
    # Make predictions
    prediction = rf_model.predict(input_data)

    # Display the prediction
    result = 'Default' if prediction[0] == 1 else 'Not Default'
    st.write('Prediction:', result)

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import joblib
from streamlit_folium import folium_static


# Load the decision tree model (assuming it's available in the same directory)
decision_tree_model = joblib.load('decision_tree_model.pkl')

town_names = {
    'ANG MO KIO': 0,
    'BEDOK': 1,
    'BISHAN': 2,
    'BUKIT BATOK': 3,
    'BUKIT MERAH': 4,
    'BUKIT PANJANG': 5,
    'BUKIT TIMAH': 6,
    'CENTRAL AREA': 7,
    'CHOA CHU KANG': 8,
    'CLEMENTI': 9,
    'GEYLANG': 10,
    'HOUGANG': 11,
    'JURONG EAST': 12,
    'JURONG WEST': 13,
    'KALLANG/WHAMPOA': 14,
    'LIM CHU KANG': 15,
    'MARINE PARADE': 16,
    'PASIR RIS': 17,
    'PUNGGOL': 18,
    'QUEENSTOWN': 19,
    'SEMBAWANG': 20,
    'SENGKANG': 21,
    'SERANGOON': 22,
    'TAMPINES': 23,
    'TOA PAYOH': 24,
    'WOODLANDS': 25,
    'YISHUN': 26
}

room_types = {
    '1 ROOM': 0,
    '2 ROOM': 1,
    '3 ROOM': 2,
    '4 ROOM': 3,
    '5 ROOM': 4,
    'EXECUTIVE': 5,
    'MULTI-GENERATION': 6
}

storey_range_values = {
    '01 TO 03': 0,
    '04 TO 06': 1,
    '07 TO 09': 2,
    '10 TO 12': 3,
    '13 TO 15': 4,
    '16 TO 18': 5,
    '19 TO 21': 6,
    '22 TO 24': 7,
    '25 TO 27': 8,
    '28 TO 30': 9,
    '31 TO 33': 10,
    '34 TO 36': 11,
    '37 TO 39': 12,
    '40 TO 42': 13,
    '43 TO 45': 14,
    '46 TO 48': 15,
    '49 TO 51': 16
}

flat_model_types = {
    '2-ROOM': 0,
    '3GEN': 1,
    'ADJOINED FLAT': 2,
    'APARTMENT': 3,
    'DBSS': 4,
    'IMPROVED': 5,
    'IMPROVED-MAISONETTE': 6,
    'MAISONETTE': 7,
    'MODEL A': 8,
    'MODEL A-MAISONETTE': 9,
    'MODEL A2': 10,
    'MULTI GENERATION': 11,
    'NEW GENERATION': 12,
    'PREMIUM APARTMENT': 13,
    'PREMIUM APARTMENT LOFT': 14,
    'PREMIUM MAISONETTE': 15,
    'SIMPLIFIED': 16,
    'STANDARD': 17,
    'TERRACE': 18,
    'TYPE S1': 19,
    'TYPE S2': 20
}

st.title('🏢 Singapore Flat Resale Price Prediction 🦁')

st.subheader('Introduction')
st.markdown('''
The objective of this project is to develop a machine learning model and deploy it as a user-friendly 
web application that predicts the resale prices of flats in Singapore. 
This predictive model will be based on historical data of resale flat transactions.
''')

st.markdown('''
### 📊 Predict the resale price using the below factors''')

# Define user input fields
town = st.selectbox('Town', options=list(town_names.keys()))
storey_range = st.selectbox('Storey Range', options=list(storey_range_values.keys()))

min_price_per_sqm = 160
max_price_per_sqm = 8000
price_options = list(range(min_price_per_sqm, max_price_per_sqm + 1, 100))
price_per_sqm = st.select_slider('Price per Square Meter', options=price_options)

flat_type = st.selectbox('Flat Type', options=list(room_types.keys()))
flat_model = st.selectbox('Flat Model', options=list(flat_model_types.keys()))

floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0.0, max_value=500.0, value=100.0)

min_age_of_property = 2
max_age_of_property = 58
age_of_property = st.number_input('Age of Property (Years)', min_value=min_age_of_property, 
                                  max_value=max_age_of_property, value=min_age_of_property)

min_year = 1966
max_year = 2022
lease_commence_date = st.number_input('Lease Commencement Year', min_value=min_year, max_value=max_year, 
                                      value=min_year)

min_current_remaining_lease = 40
max_current_remaining_lease = 97
current_remaining_lease = st.number_input('Current Remaining Lease (Years)', min_value=min_current_remaining_lease, 
                                          max_value=max_current_remaining_lease, value=min_current_remaining_lease)

min_remaining_lease = 40
max_remaining_lease = 99
remaining_lease = st.number_input('Remaining Lease (Years)', min_value=min_remaining_lease, 
                                  max_value=max_remaining_lease, value=min_remaining_lease)

min_years_holding = 0
max_years_holding = 60
years_holding = st.number_input('Years Holding (Years)', min_value=min_years_holding, 
                                max_value=max_years_holding, value=min_years_holding)

# Define a function to make predictions
def predict_price(town, flat_type, storey_range, floor_area_sqm, flat_model,
                  lease_commence_date, remaining_lease, price_per_sqm, years_holding,
                  current_remaining_lease, age_of_property):
    
    # Get the encoded values for flat type and flat model
    encoded_flat_type = room_types[flat_type]
    encoded_flat_model = flat_model_types[flat_model]
    encoded_town = town_names[town]
    encoded_storey_range = storey_range_values[storey_range]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'town': [encoded_town],
        'flat_type': [encoded_flat_type],
        'storey_range': [encoded_storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'flat_model': [encoded_flat_model],
        'lease_commence_date': [lease_commence_date], 
        'remaining_lease': [remaining_lease],
        'price_per_sqm': [price_per_sqm],     
        'years_holding': [years_holding],    
        'current_remaining_lease': [current_remaining_lease],
        'age_of_property': [age_of_property],                      
    })
    
    # Convert any Timestamp objects to strings
    for col in input_data.columns:
        if pd.api.types.is_datetime64_any_dtype(input_data[col]):
            input_data[col] = input_data[col].astype(str)

    # Prediction with decision tree model
    prediction = decision_tree_model.predict(input_data)

    # Return the predicted price
    return prediction[0]

# Get user inputs and make prediction when 'Predict' button is clicked
if st.button('Predict'):
    prediction = predict_price(town, flat_type, storey_range, floor_area_sqm, flat_model,
                               lease_commence_date, remaining_lease, price_per_sqm, years_holding,
                               current_remaining_lease, age_of_property)
    st.write('### Predicted Resale Price:', prediction)

    # Add data to a DataFrame for visualization
    vis_data = pd.DataFrame({
        'Age of Holding': list(range(min_years_holding, max_years_holding + 1)),
        'Predicted Resale Price': [predict_price(town, flat_type, storey_range, floor_area_sqm, flat_model,
                                                 lease_commence_date, remaining_lease, price_per_sqm, yh,
                                                 current_remaining_lease, age_of_property) for yh in range(min_years_holding, max_years_holding + 1)]
    })

    st.write('## Additional Visualizations')
    
    # Line chart of predicted resale price vs. age of holding
    st.scatter_chart(vis_data.set_index('Predicted Resale Price'))


# Load geospatial data (replace 'singapore_towns.geojson' with your actual file path)
geo_data = gpd.read_file('sample.geojson')

# Convert any Timestamp objects in the GeoDataFrame to strings
for col in geo_data.columns:
    if pd.api.types.is_datetime64_any_dtype(geo_data[col]):
        geo_data[col] = geo_data[col].astype(str)

# Create a Folium map centered around Singapore
m = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

# Add geospatial data to the map with tooltips and popups
for _, row in geo_data.iterrows():
    name = row['townName']  # Change 'name' to the appropriate column in your GeoJSON
    population = row['population']  # Example additional detail; change as needed
    
    # Create a tooltip with the name of the area
    tooltip = folium.Tooltip(name)
    
    # Create a popup with additional details
    popup_content = f"Name: {name}<br>Population: {population}"
    popup = folium.Popup(popup_content, max_width=300)
    
    # Add GeoJson layer with tooltips and popups
    folium.GeoJson(
        row['geometry'],
        tooltip=tooltip,
        popup=popup
    ).add_to(m)

# Display the map in the Streamlit app
folium_static(m)

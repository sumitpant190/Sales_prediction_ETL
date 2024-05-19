import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model from the pickle file
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess user input data
def preprocess_input(user_input_data):
    # Perform one-hot encoding for 'Outlet_Type' and 'Item_Type' columns
    user_input_data = pd.get_dummies(user_input_data, columns=['Outlet_Type', 'Item_Type'])

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Perform label encoding for 'Outlet_Location_Type', 'Outlet_Size', and 'Item_Fat_Content' columns
    user_input_data['Outlet_Location_Type_LabelEncoded'] = label_encoder.fit_transform(user_input_data['Outlet_Location_Type'])
    user_input_data['Outlet_Size_LabelEncoded'] = label_encoder.fit_transform(user_input_data['Outlet_Size'])
    user_input_data['Item_Fat_Content_LabelEncoded'] = label_encoder.fit_transform(user_input_data['Item_Fat_Content'])

    # Drop the original columns after label encoding
    user_input_data.drop(columns=['Outlet_Location_Type', 'Outlet_Size', 'Item_Fat_Content'], inplace=True)

    encoded_columns = [
        'Item_Visibility', 'Item_MRP', 'Item_Weight',
        'Outlet_Type_Grocery Store', 'Outlet_Type_Supermarket Type1',
        'Outlet_Type_Supermarket Type2', 'Outlet_Type_Supermarket Type3',
        'Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast',
        'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
        'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
        'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
        'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
        'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
        'Outlet_Location_Type_LabelEncoded', 'Outlet_Size_LabelEncoded',
        'Item_Fat_Content_LabelEncoded'
     ]

    # Ensure all columns from encoded_df_train are present in user_input_data
    missing_columns = set(encoded_columns) - set(user_input_data.columns)
    for col in missing_columns:
        user_input_data[col] = 0  # Add missing columns with all zeros

    # Reorder the columns to match the order in encoded_df_train
    user_input_data = user_input_data[encoded_columns]

    return user_input_data


# Function to make predictions
def predict(model, input_features):
    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features])

    # Preprocess input data
    input_df_processed = preprocess_input(input_df)

    # Make predictions
    prediction = model.predict(input_df_processed)

    return prediction

def main():
    st.title('Predict Outlet Sales')

    # Load the trained model
    model_path = r"/mnt/f/best_model.pkl"  # Update with the path to your pickle file
    model = load_model(model_path)

    # Define input fields for user input
    input_features = {}

    st.sidebar.title('Input Features')

    # Example input fields - Replace with your actual features
    input_features['Item_Weight'] = st.sidebar.number_input('Item Weight', value=0.0)
    input_features['Item_Visibility'] = st.sidebar.number_input('Item Visibility', value=0.0)
    input_features['Item_MRP'] = st.sidebar.number_input('Item MRP', value=0.0)
    input_features['Outlet_Type'] = st.sidebar.selectbox('Outlet Type', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    input_features['Item_Type'] = st.sidebar.selectbox('Item Type', ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])
    input_features['Outlet_Location_Type'] = st.sidebar.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    input_features['Outlet_Size'] = st.sidebar.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    input_features['Item_Fat_Content'] = st.sidebar.selectbox('Item Fat Content', ['Low Fat', 'Regular'])

    # Make predictions when the 'Predict' button is clicked
    if st.sidebar.button('Predict'):
        prediction = predict(model, input_features)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()

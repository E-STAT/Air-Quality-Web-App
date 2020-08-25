import streamlit as st
import pandas as pd
import pickle
from PIL import Image


#set app title
st.title("Air Quality Prediction Web App")
image = Image.open('air_quality_image.jpg')
st.image(image, caption='Air Pollution image', use_column_width= True)
#Add image

st.header("An app that predicts PM2.5")


train = pd.read_csv('Train.csv').head()
st.write("Viewing the first five observations of the data", train)


#import model
model = pickle.load(open("model_catboost.sav","rb"))


def user_input():
    """
    This function accepts input from users using sidebar and selectbox
    :return: Pandas DataFrame
    """

    u_component_of_wind_10m_above_ground = st.sidebar.slider('u_component_of_wind_10m_above_ground',-15.0, 17.00, 0.22 )
    L3_NO2_sensor_altitude = st.sidebar.slider('L3_NO2_sensor_altitude', -1.0, 1000000.0, 831329.012)
    L3_CO_cloud_height = st.sidebar.slider('L3_CO_cloud_height', -489.869019, 4999.333496, 508.449762)
    L3_NO2_NO2_column_number_density = st.sidebar.slider('L3_NO2_NO2_column_number_density', -1.00, 1.000, 0.000072)
    L3_AER_AI_sensor_altitude = st.sidebar.slider('L3_AER_AI_sensor_altitude', -1.000000, 844493.897695, 832029.566413)
    month = st.sidebar.selectbox('Months', (1, 2, 3, 4))
    L3_CO_CO_column_number_density = st.sidebar.slider('L3_CO_CO_column_number_density', -1.00, 1.00, 0.03)
    Place_ID_freq = st.sidebar.slider('Place_ID_freq', 0.00, 1.00, 0.003)


    features = {
        'u_component_of_wind_10m_above_ground': u_component_of_wind_10m_above_ground,
        'L3_NO2_sensor_altitude': L3_NO2_sensor_altitude,
        'L3_CO_cloud_height': L3_CO_cloud_height,
        'L3_NO2_NO2_column_number_density': L3_NO2_NO2_column_number_density,
        'L3_AER_AI_sensor_altitude': L3_AER_AI_sensor_altitude,
        'month': month,
        'L3_CO_CO_column_number_density': L3_CO_CO_column_number_density,
        'Place_ID_freq': Place_ID_freq,
    }


    data = pd.DataFrame(features, index = [0])

    return data

#testing
get_user_input = user_input()

#predict
prediction = model.predict(get_user_input)

st.write("The value of pm2.5 predicted is", prediction)









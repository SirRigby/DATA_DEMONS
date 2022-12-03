import streamlit as st
import joblib
import os


import numpy as np
import datetime

from PIL import Image
attrib_info="""
#### Fields:
    - Police Force 
    - Number of vehicles
    - Number of casualities
    - 2nd Road Class
    - Speed limit
    - Urban or Rural Area
"""
@st.cache(allow_output_mutation=True)
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def ml_app():
    st.subheader("Machine Learning Section")
    loaded_model=load_model("D:/appp/modelxgbnew.pkl")
    col1,col2=st.columns(2)
    with col1:
        latitude = st.number_input("Latitude",step=1e-6,format="%.6f")
        d = st.date_input("Enter Date",datetime.date(2022, 12, 2),min_value=datetime.date(2001, 1, 1))
        week = st.selectbox("Day of Week",[1,2,3,4,5,6,7])
    with col2:
        longitude = st.number_input("Longitude",step=1e-6,format="%.6f")
        road = st.number_input("1st Road Number",format="%.0f")
    encoded_result=[latitude,longitude,d.day,d.month,road,d.year,week]
    with st.expander("Results"):
        sample=np.array(encoded_result).reshape(1,-1)
        prediction = loaded_model.predict(sample)
        st.success('Predicted accident severity: '+str(prediction[0]+1))
        pred_prob = loaded_model.predict_proba(sample)
        image = Image.open('D:/appp/Classification_Report.png')
        st.image(image,caption='Classification Report')
      #  st.write("  Accuracy :"+str(prob*100)+'%'+str(probb*100)+'%')
        
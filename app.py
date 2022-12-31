import streamlit as st
import streamlit.components.v1 as stt

from eda import eda_app
from ml import ml_app

def main():
    st.title("Project")
    st.header("Accident Severity Predictor")
    menu=["Home","EDA","Predictor","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.write("")
        st.write('\n \n \n')
        st.subheader("Submission by")
       # st.subheader(" >>>> DATA_DEMONS <<<<")
        st.write("#### _**Brief Description**_")
        st.markdown("""
               - 1. Loading the Dataset using Pandas Module as a dataframe for pre-processing
               
               &nbsp;
""")
        st.markdown("""
- 2. Data Cleaning:
    
	> 1.Dropping the Null Values (No. of null values is very low as compared to the data)
    
	> 2.Dropping the columns with high correlation using pearson correlation
    
	> 3.The total number of the duplicated rows are 34155. Hence, we can drop these rows
    
	> 4.Checking the Data Trends with plots like Diagnostic Plot and Correlation Matrix
    
	> 5.Plotting the box plot for the detection of outliers
    
&nbsp;	
    

- 3. Exploratory Data Analysis:
    
	> 1.Plots such as Heatmaps and Matrix are displayed
    
	> 2.Plots like pie chart are for bivariate analysis of Data
    
	> 3.Plots such as Count Plot for analysis of categorical-ordinal Variable
    
&nbsp;	
    
- 4. Timeseries Analysis:
    
    
	> 1.Timeseries Anlysis using Plotly for Deeper insights
    
   &nbsp; 
	
- 5. Encoding and Scaling the Data

	> 1.Encoding is done using Label Encoder
    
	> 2.Scaling is Done by using Standard Scaler from the sklearn.preprocessing
    
    
&nbsp;
- 6. Performing Train Test Spilt using sklearn

&nbsp;

- 7. Using SMOTE for the treatment of Imbalanced Dataset
&nbsp;


- 8. Using the XGBoost Model for the prediction to maximise the accuracy 

&nbsp;

- 9. Using Confusion Matrix and Classification Report for the Validation and Calculation of accuracy and other parameters

&nbsp;

- 10. Creating a Pickle File to dump the model for deployment     
                    """)
    elif choice == "EDA":
        eda_app()
    elif choice == "Predictor":
        ml_app()
    else:
        st.subheader("About")
        st.write("- Piyush Kumar")
        st.write("Information Technology, sophomore")
        st.write("- Raj Chaudhary") 
        st.write("Information Technology, sophomore")
        st.write("- Vansh Kunwar Ji")
        st.write("Information Technology, sophomore")
main()

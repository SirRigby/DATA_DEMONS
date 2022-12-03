import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

@st.cache(allow_output_mutation=True)

def retrieving_data():
    df = pd.read_csv('raw_data.csv',parse_dates=['Date', 'Time'])
    #df.drop(columns=['Unnamed: 0', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Local_Authority_(Highway)', 'LSOA_of_Accident_Location'], inplace=True)
    return df

def retrieve2(df):
    df.dropna(subset=['Longitude', 'Time', 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities'], inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def retrieve3(df):
    #df.drop(columns=['Local_Authority_(District)'], axis=1, inplace=True)
    return df

def retrieve4(df):
    #df.drop('Accident_Index',axis=1,inplace=True)
    return df

def eda_app():
    df=retrieving_data()
    st.subheader("EDA Section")
    submenu=st.sidebar.selectbox("Submenu",["Descriptive","Plots"])
    if submenu=="Descriptive":
        st.write("""### Data Sample""")
        st.dataframe(df.sample(30))
        
        with st.expander("Data Summary"):
            st.write("No. of rows : ",df.shape[0])
            st.write("No. of columns",df.shape[1])
        with st.expander("Data Types Summary"):
            st.dataframe(df.dtypes)
        with st.expander("Descriptive Summary"):
            st.dataframe(df.describe(include=np.object))
            st.dataframe(df.describe(include=np.number))
    else:
        df=retrieve2(df)
        numerical_data = df.select_dtypes(include='number')
        num_cols = numerical_data.columns
        st.subheader("Plots")
        with st.expander("Box plots for outliers"):
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(20, 50))
            fig.subplots_adjust(right=1.5)
            for plot in range(1, len(num_cols)+1):
                plt.subplot(6, 4, plot)
                sns.boxplot(y=df[num_cols[plot-1]])
            st.pyplot(fig)
        with st.expander("Diagnostic Plots"):
            def diagnostic_plot(data, col):
                fig = plt.figure(figsize=(20, 5))
                fig.subplots_adjust(right=1.5)
    
                plt.subplot(1, 3, 1)
                sns.distplot(data[col], kde=True, color='teal')
                plt.title('Histogram')
    
                plt.subplot(1, 3, 2)
                stats.probplot(data[col], dist='norm', fit=True, plot=plt)
                plt.title('Q-Q Plot')
                plt.subplot(1, 3, 3)
                sns.boxplot(data[col],color='teal')
                plt.title('Box Plot')
    
                st.pyplot(fig)
    
            dist_lst = ['Police_Force', 'Accident_Severity','Number_of_Vehicles', 'Number_of_Casualties', 'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number','Speed_limit', '2nd_Road_Class', '2nd_Road_Number','Urban_or_Rural_Area']

            for col in dist_lst:
                diagnostic_plot(df, col)
        with st.expander("Correlation matrix"):
            fig=plt.figure(figsize = (15,10))
            corr = df.corr(method='spearman')
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
            cormat.set_title('Correlation Matrix')
            st.pyplot(fig)
        df=retrieve3(df)
        with st.expander("Pie Chart (Presence of police officer)"):
            def pie_chart(data, col):
                x = data[col].value_counts().values
                fig=plt.figure(figsize=(7, 6))
                plt.pie(x, center=(0, 0), radius=1.5, labels=data[col].unique(),autopct='%1.1f%%', pctdistance=0.5)
                plt.axis('equal')
                st.pyplot(fig)
            pie_lst = ['Did_Police_Officer_Attend_Scene_of_Accident']
            for col in pie_lst:
                pie_chart(df, col)
        with st.expander("Count plot of categorical variable Type 1"):
            def cnt_plot(data, col):
                fig=plt.figure(figsize=(15, 7))
                ax1 = sns.countplot(x=col, data=data)
                for p in ax1.patches:
                    ax1.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1), ha='center')
                st.pyplot(fig)
                print('\n')

            cnt_lst1 = ['Road_Type', 'Junction_Control','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions']

            for col in cnt_lst1:
                cnt_plot(df, col)
        with st.expander("Count plot of categorical variable Type 2"):
            def cnt_plot(data, col):
                fig=plt.figure(figsize=(10, 7))
                sns.countplot(y=col, data=data)
                st.pyplot(fig)

                print('\n')
  
            cnt_lst2 = ['Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions','Weather_Conditions','Special_Conditions_at_Site', 'Carriageway_Hazards']
            for col in cnt_lst2:
                cnt_plot(df, col)
        df['Urban_or_Rural_Area'].replace(3, 1, inplace=True)
        with st.expander('Time Series Analysis'):
            dt1 = df.groupby('Date')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
            fig = px.line(dt1, x='Date', y='No. of Accidents',
            labels={'index': 'Date', 'value': 'No. of Accidents'})
            st.plotly_chart(fig)
            #dt2 = df.groupby('Year')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
            #fig = px.line(dt2, x='Year', y='No. of Accidents',labels={'index': 'Year', 'value': 'No. of Accidents'})
            #st.plotly_chart(fig)
            dt3 = df.groupby('Day_of_Week')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})
            fig = px.line(dt3, x='Day_of_Week', y='No. of Accidents',labels={'index': 'Day_of_Week', 'value': 'No. of Accidents'})
            st.plotly_chart(fig)
        df=retrieve4(df)
        with st.expander('Heat Map'):
            fig=plt.figure(figsize=(18, 5))
            sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
            st.pyplot(fig)
        with st.expander('Correlation Plot'):
            X = df.drop(columns=['Accident_Severity'], axis=1)
            fig=plt.figure(figsize=(8, 10))
            X.corrwith(df['Accident_Severity']).plot(kind='barh',title="Correlation with 'Convert' column -")
            st.pyplot(fig)
        with st.expander('Confusion Matrix'):
            image = Image.open('confusion_matrix.png')
            st.image(image)

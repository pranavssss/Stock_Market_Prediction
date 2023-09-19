import urllib.request
import requests
import streamlit as st
import plotly.express as px
import sklearn
import pickle
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie


st.set_page_config(page_title="Stock Prediction", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_book = load_lottieurl("https://lottie.host/ab9593bf-b280-4574-9cdc-7e6d5ad1026d/jJGiycZrTW.json")
st_lottie(lottie_book, speed=1, height=300, key="initial")


# Example: Display an image
#st.image("/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/logo.png", use_column_width=True)
col_1,col_2,col_3 = st.columns(3)

with col_2:
    st.title("STOCKER : STOCK PREDICTION MODEL USING SOCIAL MEDIA SENTIMENT ANALYSIS")
    st.text("")
# Add your content here
st.write("")


stocks = ('GOOGLE', 'APPLE' , 'AMAZON' , 'MICROSOFT' , 'META' , 'NETFLIX' , 'INTEL' , 'ZSCALER' , 'PAYPAL' , 'TESLA')
selected_stock = st.selectbox('Select stock for prediction', stocks)

# Define a dictionary mapping image options to their respective image URLs or file paths
image_dict = {
    
    "GOOGLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/GOOGLE/train_plot.png",
    "APPLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/APPLE/train_plot.png",
    "AMAZON": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/AMAZON/train_plot.png",
    "MICROSOFT": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/MICROSOFT/train_plot.png",
    "META": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/META/train_plot.png",
    "NETFLIX": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/NETFLIX/train_plot.png",
    "INTEL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/INTEL/train_plot.png",
    "ZSCALER": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/ZSCALER/train_plot.png",
    "PAYPAL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/PAYPAL/train_plot.png",
    "TESLA": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/TESLA/train_plot.png"
}

logo_dict = {
    
    "GOOGLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/GOOGLE/logo.png",
    "APPLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/APPLE/logo.png",
    "AMAZON": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/AMAZON/logo.png",
    "MICROSOFT": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/MICROSOFT/logo.png",
    "META": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/META/logo.png",
    "NETFLIX": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/NETFLIX/logo.png",
    "INTEL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/INTEL/logo.png",
    "ZSCALER": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/ZSCALER/logo.png",
    "PAYPAL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/PAYPAL/logo.png",
    "TESLA": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/TESLA/logo.png"
}

graph_dict = {
    
    "GOOGLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/GOOGLE/Finaldata_with_Fourier.csv",
    "APPLE": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/APPLE/Finaldata_with_Fourier.csv",
    "AMAZON": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/AMAZON/Finaldata_with_Fourier.csv",
    "MICROSOFT": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/MICROSOFT/Finaldata_with_Fourier.csv",
    "META": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/META/Finaldata_with_Fourier.csv",
    "NETFLIX": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/NETFLIX/Finaldata_with_Fourier.csv",
    "INTEL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/INTEL/Finaldata_with_Fourier.csv",
    "ZSCALER": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/ZSCALER/Finaldata_with_Fourier.csv",
    "PAYPAL": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/PAYPAL/Finaldata_with_Fourier.csv",
    "TESLA": "/Users/laikhuramkorouhanbakhuman/Downloads/FINAL_PROJECT/PROJECT/TESLA/Finaldata_with_Fourier.csv"
}

col1, col2, col3, col4, col5 = st.columns((3,0.1,3,0.1,3))

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1))

# Display the selected image
if st.button('ANALYSE'):
    
    if selected_stock in logo_dict:

        with col3:
            
            st.text("")
            logo_path = logo_dict[selected_stock]
            st.image(logo_path, use_column_width=True, width=200)
            
        
    if selected_stock in graph_dict:
    
        with row3_1:
        
            st.text("")
            st.subheader("{} STOCK PRICING.".format(selected_stock))
            graph_path = graph_dict[selected_stock]
            df = pd.read_csv(graph_path)
            fig = px.line(
                df,
                x="Date",
                y="Close",
                color_discrete_sequence=["#9EE6CF"],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    if selected_stock in graph_dict:
    
        with row3_2:
        
            st.text("")
            st.subheader("TECHNICAL INDICATORS FOR {}.".format(selected_stock))
            graph_path = graph_dict[selected_stock]
            df = pd.read_csv(graph_path)
            fig = px.line(
                df,
                x="Date",
                y=["Close","MA7","MA21","upper_band","lower_band"],
                color_discrete_sequence=["#9EE6CF","yellow","pink","blue","red"]
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            

    if selected_stock in image_dict:

            st.text("")
            st.subheader("REAL VS PREDICTED GRAPH FOR {} STOCK.".format(selected_stock))
            image_path = image_dict[selected_stock]
            st.image(image_path, caption=selected_stock, use_column_width=True)

    else:
        st.write("No Stock selected")

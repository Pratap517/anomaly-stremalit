# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:47:12 2021

@author: Admin
"""

import streamlit as st
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest



def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "outputanomaly_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)




def main():
    global p,data
    
    st.header("Anomaly Detection")
    #st.write("Upload CSV ")
    #path=st.file_uploader("")
    #st.write(path)
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
       df = pd.read_csv(data_file)
       st.dataframe(df.head())
       st.write("total observations", df.shape)
    
       #percent=st.number_input('enter the percentage',0.1,1.0,0.1)
       percent=0.15
       
       #outputdata=anamoly(data_file,percent)
       if percent is not None:
           data = df.copy()
           p=percent
       ok=st.button("click here to generate anomaly file")
       if ok:
           for col in data.columns:
               
               if data[col].dtype == "object":
                  
                  le = LabelEncoder()
                  data[col].fillna("None", inplace=True)
                  le.fit(list(data[col].astype(str).values))
                  data[col] = le.transform(list(data[col].astype(str).values))
               else:
                  data[col].fillna(0, inplace=True) 
           model = IsolationForest(contamination=p, n_estimators=1000)
           model.fit(data) 
           df["iforest"] = pd.Series(model.predict(data))
           df["iforest"] = df["iforest"].map({1: 0, -1: 1})
           st.write("the shape of output file",df.shape)
           st.write("number of anomalies detected",df["iforest"].value_counts())
           csv_downloader(df)
                  
       


    
    

     



if __name__ == '__main__':
	main()
    
    
    
#outputdata=anamoly(data_file,percent)   